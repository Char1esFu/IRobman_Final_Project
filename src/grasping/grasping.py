import numpy as np
from typing import Tuple, Sequence, Optional, Any
import open3d as o3d
import pybullet as p  # Import pybullet for visualization


class GraspGeneration:
    def __init__(self):
        pass

    def sample_grasps(
        self,
        center_point: np.ndarray,
        num_grasps: int,
        sim = None,
        radius: float = 0.1,  # Keep parameter but not used
        rotation_matrix: np.ndarray = None,
        min_point_rotated: np.ndarray = None,
        max_point_rotated: np.ndarray = None,
        center_rotated: np.ndarray = None
    ) -> Sequence[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate multiple random grasp poses within the bounding box.

        Parameters:
            center_point: Centroid coordinates of the point cloud
            num_grasps: Number of random grasp poses to generate
            sim: Simulation object
            radius: Maximum distance offset (kept but not used)
            rotation_matrix: OBB rotation matrix (from OBB coordinate system to world coordinate system)
            min_point_rotated: Minimum point of OBB in rotated coordinate system
            max_point_rotated: Maximum point of OBB in rotated coordinate system
            center_rotated: Origin of the OBB rotated coordinate system in world coordinates

        Returns:
            list: List of rotation matrices and translation vectors
        """
        if rotation_matrix is None or min_point_rotated is None or max_point_rotated is None or center_rotated is None:
            raise ValueError("Rotation matrix and OBB coordinates must be provided")

        grasp_poses_list = []
        table_height = sim.robot.pos[2] # Robot base height
        
        # Calculate the size of OBB in three dimensions
        obb_dims = max_point_rotated - min_point_rotated
        
        # Determine the length sorting of dimensions
        dims_with_idx = [(obb_dims[i], i) for i in range(3)]
        dims_with_idx.sort()  # Sort by dimension length in ascending order
        
        # Get the shortest edge index and second shortest edge index
        shortest_axis_idx = dims_with_idx[0][1]  # Coordinate axis index corresponding to the shortest edge
        second_shortest_idx = dims_with_idx[1][1]  # Coordinate axis index corresponding to the second shortest edge
        longest_axis_idx = dims_with_idx[2][1]  # Coordinate axis index corresponding to the longest edge

        # Determine if the object is tall enough to apply the complete bounding box alignment strategy
        # Set height threshold (unit: meters)
        height_threshold = 0.35  # 35 centimeters
        
        # Calculate the vertical height of the object in the world coordinate system (z-axis direction)
        object_z_height = obb_dims[2]  # Assume the 3rd dimension of the object coordinate system corresponds to the vertical direction
        
        # Determine if the vertical height exceeds the threshold
        is_tall_object = object_z_height > height_threshold

        # For flat objects, pre-calculate the centroid coordinates in the OBB coordinate system for subsequent sampling
        if not is_tall_object:
            # Convert centroid from world coordinate system to OBB coordinate system
            if center_point is not None:
                # center_point is the centroid in world coordinate system, needs to be converted to OBB coordinate system
                center_relative_to_obb = center_point - center_rotated
                center_in_obb = np.dot(center_relative_to_obb, rotation_matrix)
            else:
                # If centroid is not provided, use the center of OBB
                center_in_obb = (min_point_rotated + max_point_rotated) / 2
        
        for idx in range(num_grasps):
            # Sample inside OBB, using normal distribution for the shortest edge
            rotated_coords = [0, 0, 0]
            
            # Sample the shortest edge using normal distribution, dense in the middle
            min_val = min_point_rotated[shortest_axis_idx]
            max_val = max_point_rotated[shortest_axis_idx]
            
            # Mean is the midpoint of the edge length
            mean = (min_val + max_val) / 2
            # Standard deviation set to 1/6 of the edge length, so the edge length range is about ±3 standard deviations, covering 99.7% of the normal distribution
            std = (max_val - min_val) / 6
            
            # Use truncated normal distribution to ensure values are within boundaries
            while True:
                sample = np.random.normal(mean, std)
                if min_val <= sample <= max_val:
                    rotated_coords[shortest_axis_idx] = sample
                    break
            
            # Determine if it's a flat object (vertical downward grasping situation)
            if not is_tall_object:
                # In front view, another edge besides the shortest edge (could be the longest or second shortest)
                # Determine which axis is the vertical axis
                vertical_axis_idx = None
                axes = [rotation_matrix[:, i] for i in range(3)]
                
                # Calculate the dot product of each axis with the world coordinate system Z-axis, find the axis closest to the vertical direction
                z_world = np.array([0, 0, 1])
                z_dots = [abs(np.dot(axis, z_world)) for axis in axes]
                vertical_axis_idx = np.argmax(z_dots)
                
                # Find the two axes on the horizontal plane (except the vertical axis)
                horizontal_axes = [i for i in range(3) if i != vertical_axis_idx]
                
                # Now we have two horizontal axes, one of which is the shortest edge
                # The other is the "other edge" we're looking for
                other_axis_idx = horizontal_axes[0] if horizontal_axes[0] != shortest_axis_idx else horizontal_axes[1]
                
                # Sample this "other edge" using normal distribution centered at the centroid
                min_other = min_point_rotated[other_axis_idx]
                max_other = max_point_rotated[other_axis_idx]
                
                # Use the projection of the centroid on this axis as the mean
                mean_other = center_in_obb[other_axis_idx]
                # If the centroid projection is outside the bounding box range, use the midpoint of the bounding box as the mean
                if mean_other < min_other or mean_other > max_other:
                    mean_other = (min_other + max_other) / 2
                
                # Standard deviation set to 1/6 of the edge length
                std_other = (max_other - min_other) / 6
                
                # Sample using truncated normal distribution
                while True:
                    sample = np.random.normal(mean_other, std_other)
                    if min_other <= sample <= max_other:
                        rotated_coords[other_axis_idx] = sample
                        break
                
                # Vertical axis (Z-axis) uses uniform distribution or distribution biased towards the top
                # Since it's vertical downward grasping, we might prefer to start grasping from near the top of the object
                min_z = min_point_rotated[vertical_axis_idx]
                max_z = max_point_rotated[vertical_axis_idx]
                # Top-biased sampling (using Beta distribution or other biased distribution)
                # Simplified here as uniform distribution, but closer to the top
                top_bias = 0.7  # Degree of bias towards the top, 0.5 is uniform, larger is closer to the top
                z_sample = min_z + (max_z - min_z) * (1 - np.random.beta(1, top_bias))
                rotated_coords[vertical_axis_idx] = z_sample
            else:
                # For tall objects, the remaining two axes still use uniform distribution
                rotated_coords[second_shortest_idx] = np.random.uniform(
                    min_point_rotated[second_shortest_idx],
                    max_point_rotated[second_shortest_idx]
                )
                rotated_coords[longest_axis_idx] = np.random.uniform(
                    min_point_rotated[longest_axis_idx], 
                    max_point_rotated[longest_axis_idx]
                )
            
            grasp_center_rotated = np.array(rotated_coords)
            
            # Transform the sampling point from rotated coordinate system back to world coordinate system
            grasp_center = np.dot(grasp_center_rotated, rotation_matrix.T) + center_rotated
            
            # Ensure grasp point is not lower than the table height
            grasp_center[2] = max(grasp_center[2], table_height)
            print(f"grasp_center: {grasp_center}")

            # Determine grasp pose based on OBB
            # 1. Calculate the size of OBB in three dimensions
            obb_dims = max_point_rotated - min_point_rotated
            
            # 2. Determine the length sorting of dimensions
            dims_with_idx = [(obb_dims[i], i) for i in range(3)]
            dims_with_idx.sort()  # Sort by dimension length in ascending order
            
            # Get the shortest edge index and second shortest edge index
            shortest_axis_idx = dims_with_idx[0][1]  # Coordinate axis index corresponding to the shortest edge
            second_shortest_idx = dims_with_idx[1][1]  # Coordinate axis index corresponding to the second shortest edge
            longest_axis_idx = dims_with_idx[2][1]  # Coordinate axis index corresponding to the longest edge
            
            # 3. Extract three coordinate axis directions from rotation_matrix
            axes = [rotation_matrix[:, i] for i in range(3)]  # Three axis directions of OBB
            
            # Determine if the object is tall enough to apply the complete bounding box alignment strategy
            # Set height threshold (unit: meters)
            height_threshold = 0.35  # 35 centimeters
            
            # Calculate the vertical height of the object in the world coordinate system (z-axis direction), not the longest edge
            # The z-axis height of the object is defined as the dimension of the bounding box in the z direction
            object_z_height = obb_dims[2]  # Assume the 3rd dimension of the object coordinate system corresponds to the vertical direction
            
            # Determine if the vertical height exceeds the threshold
            is_tall_object = object_z_height > height_threshold
            
            # 4. Construct new rotation matrix
            if is_tall_object:
                # For tall objects, use the complete bounding box alignment strategy
                # Gripper X-axis (fingertip opening direction) corresponds to the shortest edge
                grasp_x_axis = axes[shortest_axis_idx]
                # Gripper Y-axis (fingertip extension direction) corresponds to the second shortest edge
                grasp_y_axis = axes[second_shortest_idx]
                # Gripper Z-axis determined by cross product
                grasp_z_axis = np.cross(grasp_x_axis, grasp_y_axis)
                
                # Ensure coordinate system is right-handed
                dot_product = np.dot(grasp_z_axis, axes[longest_axis_idx])
                if dot_product < 0:
                    grasp_z_axis = -grasp_z_axis
                
                # Construct rotation matrix based on object principal axes
                R = np.column_stack((grasp_x_axis, grasp_y_axis, grasp_z_axis))
            else:
                # For flat objects, use simplified strategy
                # Align Z-axis with the world coordinate system's Z-axis, pointing upward
                grasp_z_axis = np.array([0, 0, 1])  # Vertical upward
                
                # Try to get the X-axis (fingertip opening direction) from the shortest edge of the object
                # But first ensure this direction is not parallel to the Z-axis
                candidate_x = axes[shortest_axis_idx]
                dot_xz = np.dot(candidate_x, grasp_z_axis)
                
                if abs(dot_xz) > 0.9:  # Shortest edge is almost vertical, use second shortest edge instead
                    candidate_x = axes[second_shortest_idx]
                    dot_xz = np.dot(candidate_x, grasp_z_axis)
                    
                    if abs(dot_xz) > 0.9:  # Second shortest edge is also almost vertical, use fixed lateral direction
                        grasp_x_axis = np.array([1, 0, 0])
                    else:
                        # Project second shortest edge to horizontal plane as X-axis
                        grasp_x_axis = candidate_x - grasp_z_axis * dot_xz
                else:
                    # Project shortest edge to horizontal plane as X-axis
                    grasp_x_axis = candidate_x - grasp_z_axis * dot_xz
                
                # Ensure X-axis is non-zero and normalized
                if np.linalg.norm(grasp_x_axis) < 1e-6:
                    grasp_x_axis = np.array([1, 0, 0])
                else:
                    grasp_x_axis = grasp_x_axis / np.linalg.norm(grasp_x_axis)
                
                # Calculate Y-axis by cross product (ensure orthogonality)
                grasp_y_axis = np.cross(grasp_z_axis, grasp_x_axis)
                grasp_y_axis = grasp_y_axis / np.linalg.norm(grasp_y_axis)
                
                # Recalculate X-axis to ensure strict orthogonality
                grasp_x_axis = np.cross(grasp_y_axis, grasp_z_axis)
                grasp_x_axis = grasp_x_axis / np.linalg.norm(grasp_x_axis)
                
                # Rotate gripper so Y-axis points in negative Z direction (downward)
                # Gripper coordinate system: X-fingertip opening direction, Y-finger extension direction, Z-palm direction
                # Create rotation matrix: [-x, -z, -y], making fingers point downward
                R_adjust = np.array([
                    [1, 0, 0],
                    [0, 0, -1],
                    [0, 1, 0]
                ])
                
                # Construct our base rotation matrix
                R_base = np.column_stack((grasp_x_axis, grasp_y_axis, grasp_z_axis))
                
                # Apply adjustment rotation
                R = R_base @ R_adjust
            
            # Print grasp strategy information
            if idx == 0:  # Print only once
                if is_tall_object:
                    print(f"Object z-axis height ({object_z_height:.3f}m) exceeds threshold ({height_threshold:.3f}m), using complete bounding box alignment strategy")
                else:
                    print(f"Object z-axis height ({object_z_height:.3f}m) below threshold ({height_threshold:.3f}m), using simplified grasping strategy (fingertips along shortest edge, grasping from above)")

            assert grasp_center.shape == (3,)
            grasp_poses_list.append((R, grasp_center))

        return grasp_poses_list
    

    def check_grasp_collision(
        self,
        grasp_meshes: Sequence[o3d.geometry.TriangleMesh],
        object_mesh: o3d.geometry.TriangleMesh = None,
        object_pcd = None,
        num_colisions: int = 10,
        tolerance: float = 0.00001) -> bool:
        """
        Check if there is a collision between the gripper pose and the target object using point cloud sampling method.

        Parameters:
            grasp_meshes: List of mesh geometries representing gripper components
            object_mesh: Triangle mesh of the target object (optional)
            object_pcd: Point cloud of the target object (optional)
            num_colisions: Threshold number of points to determine collision
            tolerance: Distance threshold for determining collision (meters)

        Returns:
            bool: True if collision is detected between the gripper and the object, False otherwise
        """
        # Combine gripper meshes
        combined_gripper = o3d.geometry.TriangleMesh()
        for mesh in grasp_meshes:
            combined_gripper += mesh

        # Sample points from mesh
        num_points = 5000  # Number of points for subsampling both meshes
        gripper_pcl = combined_gripper.sample_points_uniformly(number_of_points=num_points)
        
        # Determine which object representation to use
        if object_mesh is not None:
            object_pcl = object_mesh.sample_points_uniformly(number_of_points=num_points)
        elif object_pcd is not None:
            object_pcl = object_pcd
        else:
            raise ValueError("Must provide at least one parameter from object_mesh or object_pcd")

        # Build KD tree for object points
        is_collision = False
        object_kd_tree = o3d.geometry.KDTreeFlann(object_pcl)
        collision_count = 0
        for point in gripper_pcl.points:
            [_, idx, distance] = object_kd_tree.search_knn_vector_3d(point, 1)
            if distance[0] < tolerance:
                collision_count += 1
                if collision_count >= num_colisions:
                    return True  # Collision detected

        return is_collision
    
    def grasp_dist_filter(self,
                        center_grasp: np.ndarray,
                        mesh_center: np.ndarray,
                        tolerance: float = 0.05)->bool:
        is_within_range = False
        #######################TODO#######################
        if np.linalg.norm(center_grasp - mesh_center) < tolerance:
            is_within_range = True
        ##################################################
        return is_within_range
    

    def check_grasp_containment(
        self,
        left_finger_center: np.ndarray,
        right_finger_center: np.ndarray,
        finger_length: float,
        object_pcd: o3d.geometry.PointCloud,
        num_rays: int,
        rotation_matrix: np.ndarray, # rotation-mat
        visualize_rays: bool = False  # Whether to visualize rays in PyBullet
    ) -> Tuple[bool, float, float]:
        """
        Checks if any line between the gripper fingers intersects with the object mesh.

        Args:
            left_finger_center: Center of Left finger of grasp
            right_finger_center: Center of Right finger of grasp
            finger_length: Finger Length of the gripper.
            object_pcd: Point Cloud of the target object
            num_rays: Number of rays to cast
            rotation_matrix: Rotation matrix for the grasp
            visualize_rays: Whether to visualize rays in PyBullet

        Returns:
            tuple[bool, float, float]: 
            - intersection_exists: True if any line between fingers intersects object
            - containment_ratio: Ratio of rays that hit the object
            - intersection_depth: Depth of deepest intersection point
        """
        left_center = np.asarray(left_finger_center)
        right_center = np.asarray(right_finger_center)

        intersections = []
        # Check for intersections between corresponding points
        object_tree = o3d.geometry.KDTreeFlann(object_pcd)

        # Calculate the height and bounding box of the object
        points = np.asarray(object_pcd.points)
        min_point = np.min(points, axis=0)
        max_point = np.max(points, axis=0)
        object_height = max_point[2] - min_point[2]
        object_center = (min_point + max_point) / 2
        
        print(f"Object height: {object_height:.4f}m")
        print(f"Object center point: {object_center}")

        obj_triangle_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd=object_pcd, 
                                                                                          alpha=0.016)
        # I just tuned alpha till I got a complete mesh with no holes, which had the best fidelity to the shape from the pcd
        
        obj_triangle_mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(obj_triangle_mesh)
        scene = o3d.t.geometry.RaycastingScene()
        obj_id = scene.add_triangles(obj_triangle_mesh_t)

        # visualize_3d_objs([obj_triangle_mesh])

        # now to know which direction to cast the rays towards, I added another coordinate 
        #       frame in the cell above question 1 in this task (task 2)
        # As shown in the coordinate frame, the fingers' tips begin at the at y=0, z=0 line, 
        # while the rest of the fingers extend along the +y axis

        hand_width = np.linalg.norm(left_center-right_center)
        finger_vec = np.array([0, finger_length, 0])
        ray_direction = (left_center - right_center)/hand_width
        
        # Store ray start and end points for visualization
        ray_start_points = []
        ray_end_points = []
        
        # ===== Calculate gripper width direction =====
        print("Calculating gripper width direction...")
        # Calculate vector in gripper width direction (perpendicular to both ray_direction and finger_vec)
        # First calculate finger_vec direction in world coordinates
        world_finger_vec = rotation_matrix.dot(finger_vec)
        # Calculate width direction vector (cross product gives vector perpendicular to both vectors)
        width_direction = np.cross(ray_direction, world_finger_vec)
        # Normalize
        width_direction = width_direction / np.linalg.norm(width_direction)
        
        # Define width direction parameters
        width_planes = 1  # Number of planes on each side in width direction
        width_offset = 0.015  # Offset between planes (meters)
        
        # ===== Generate multiple parallel ray planes =====
        print("Generating multiple parallel ray planes...")
        # Central plane (original plane)
        rays = []
        contained = False
        rays_hit = 0
        
        # Parallel planes on both sides in width direction
        for plane in range(1, width_planes + 1):
            # Calculate current plane offset
            current_offset = width_offset * plane
            
            # Right side plane
            for i in range(num_rays):
                # Calculate sampling point along length direction, and offset in width direction
                right_point = right_center + rotation_matrix.dot((i/num_rays)*finger_vec) + width_direction * current_offset
                # Add ray from right offset point to left offset point
                rays.append([np.concatenate([right_point, ray_direction])])
                
                # Store ray start and end points for visualization - using actual finger width
                ray_start_points.append(right_point)
                ray_end_points.append(right_point + ray_direction * hand_width)
            
            # Left side plane
            for i in range(num_rays):
                # Calculate sampling point along length direction, and offset in width direction
                right_point = right_center + rotation_matrix.dot((i/num_rays)*finger_vec) - width_direction * current_offset
                # Add ray from right offset point to left offset point
                rays.append([np.concatenate([right_point, ray_direction])])
                
                # Store ray start and end points for visualization - using actual finger width
                ray_start_points.append(right_point)
                ray_end_points.append(right_point + ray_direction * hand_width)
        
        print(f"Total of {len(rays)} rays generated")
        
        # Visualize rays in PyBullet
        debug_lines = []
        if visualize_rays:
            print("Visualizing rays in PyBullet...")
            for start, end in zip(ray_start_points, ray_end_points):
                line_id = p.addUserDebugLine(
                    start.tolist(), 
                    end.tolist(), 
                    lineColorRGB=[1, 0, 0],  # Red
                    lineWidth=1,
                    lifeTime=5  # Disappear after 5 seconds
                )
                debug_lines.append(line_id)
        
        # Execute ray casting
        rays_t = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
        ans = scene.cast_rays(rays_t)
        
        # Process ray casting results
        rays_hit = 0
        max_interception_depth = o3d.core.Tensor([0.0], dtype=o3d.core.Dtype.Float32)
        rays_from_left = []
        
        # Track hits for left and right side ray planes
        left_side_hit = False
        right_side_hit = False
        
        # Calculate number of rays per plane
        rays_per_plane = num_rays
        
        # Process results for all rays
        print("Processing ray casting results...")
        for idx, hit_point in enumerate(ans['t_hit']):
            # Use actual finger width to determine if ray hit the object
            if hit_point[0] < hand_width:
                rays_hit += 1
                
                # Determine if ray belongs to left or right side plane
                total_rays_count = len(rays)
                half_rays_count = total_rays_count // 2
                
                if idx < half_rays_count:
                    # Ray from right side plane
                    right_side_hit = True
                else:
                    # Ray from left side plane
                    left_side_hit = True
                
                # Only calculate depth for rays in the center plane (original plane)
                if idx < num_rays:
                    left_new_center = left_center + rotation_matrix.dot((idx/num_rays)*finger_vec)
                    rays_from_left.append([np.concatenate([left_new_center, -ray_direction])])
        
        # Only consider contained when both left and right side planes have at least one ray hit
        contained = left_side_hit and right_side_hit
        
        containment_ratio = 0.0
        if contained:
            # Process rays from left side (only for center plane)
            if rays_from_left:
                rays_t = o3d.core.Tensor(rays_from_left, dtype=o3d.core.Dtype.Float32)
                ans_left = scene.cast_rays(rays_t)
                
                for idx, hitpoint in enumerate(ans['t_hit']):
                    if idx < num_rays:  # Only process rays in center plane
                        left_idx = 0
                        # Calculate interception depth using actual finger width
                        if hitpoint[0] < hand_width: 
                            interception_depth = hand_width - ans_left['t_hit'][0].item() - hitpoint[0].item()
                            max_interception_depth = max(max_interception_depth, interception_depth)
                            left_idx += 1

        print(f"the max interception depth is {max_interception_depth}")
        # Calculate overall ray hit ratio
        total_rays = len(rays)
        containment_ratio = rays_hit / total_rays
        print(f"Ray hit ratio: {containment_ratio:.4f} ({rays_hit}/{total_rays})")
        
        intersections.append(contained)
        # intersections.append(max_interception_depth[0])
        # return contained, containment_ratio

        # Calculate distance from grasp center to object center
        grasp_center = (left_center + right_center) / 2
        
        # Calculate total distance in 3D space
        distance_to_center = np.linalg.norm(grasp_center - object_center)
        
        # Calculate distance only in x-y plane (horizontal distance)
        horizontal_distance = np.linalg.norm(grasp_center[:2] - object_center[:2])
        
        # Calculate distance score (closer distance gives higher score)
        center_score = np.exp(-distance_to_center**2 / (2 * 0.05**2))
        
        # Calculate horizontal distance score (closer horizontal distance gives higher score)
        horizontal_score = np.exp(-horizontal_distance**2 / (2 * 0.03**2))
        
        # Incorporate both distance scores into final quality score, giving higher weight to horizontal distance
        final_quality = containment_ratio * (1 + center_score + 1.5 * horizontal_score)
        
        print(f"Grasp center: {grasp_center}")
        print(f"Horizontal distance: {horizontal_distance:.4f}m, Horizontal score: {horizontal_score:.4f}")
        print(f"Total distance: {distance_to_center:.4f}m, Total distance score: {center_score:.4f}")
        print(f"Final quality score: {final_quality:.4f}")
        
        return any(intersections), final_quality, max_interception_depth.item()

    def visualize_grasp_poses(self, 
                             pose1_pos, 
                             pose1_orn, 
                             pose2_pos, 
                             pose2_orn, 
                             axis_length=0.1):
        """
        Visualize grasp pose coordinate frames in PyBullet
        
        Parameters:
            pose1_pos: Pre-grasp position
            pose1_orn: Pre-grasp orientation (quaternion)
            pose2_pos: Final grasp position
            pose2_orn: Final grasp orientation (quaternion)
            axis_length: Coordinate axis length
        """
        # Get rotation matrix from quaternion
        pose1_rot = np.array(p.getMatrixFromQuaternion(pose1_orn)).reshape(3, 3)
        pose2_rot = np.array(p.getMatrixFromQuaternion(pose2_orn)).reshape(3, 3)
        
        # Extract direction vectors for each axis
        pose1_x_axis = pose1_rot[:, 0] * axis_length
        pose1_y_axis = pose1_rot[:, 1] * axis_length
        pose1_z_axis = pose1_rot[:, 2] * axis_length
        
        pose2_x_axis = pose2_rot[:, 0] * axis_length
        pose2_y_axis = pose2_rot[:, 1] * axis_length
        pose2_z_axis = pose2_rot[:, 2] * axis_length
        
        # Visualize Pose 1 coordinate axes
        p.addUserDebugLine(pose1_pos, pose1_pos + pose1_x_axis, [1, 0, 0], 3, 0)  # X-axis - Red
        p.addUserDebugLine(pose1_pos, pose1_pos + pose1_y_axis, [0, 1, 0], 3, 0)  # Y-axis - Green
        p.addUserDebugLine(pose1_pos, pose1_pos + pose1_z_axis, [0, 0, 1], 3, 0)  # Z-axis - Blue
        
        # Visualize Pose 2 coordinate axes
        p.addUserDebugLine(pose2_pos, pose2_pos + pose2_x_axis, [1, 0, 0], 3, 0)  # X-axis - Red
        p.addUserDebugLine(pose2_pos, pose2_pos + pose2_y_axis, [0, 1, 0], 3, 0)  # Y-axis - Green
        p.addUserDebugLine(pose2_pos, pose2_pos + pose2_z_axis, [0, 0, 1], 3, 0)  # Z-axis - Blue
        
        # Add text labels
        p.addUserDebugText("Pose 1", pose1_pos + [0, 0, 0.05], [1, 1, 1], 1.5)
        p.addUserDebugText("Pose 2", pose2_pos + [0, 0, 0.05], [1, 1, 1], 1.5)

class GraspExecution:
    """Robot grasping execution class, responsible for planning and executing complete grasping actions"""
    
    def __init__(self, sim):
        """
        Initialize grasping executor
        
        Parameters:
            sim: Simulation environment object
        """
        self.sim = sim
        from src.ik_solver import DifferentialIKSolver
        self.ik_solver = DifferentialIKSolver(sim.robot.id, sim.robot.ee_idx, damping=0.05)
        from src.path_planning.simple_planning import SimpleTrajectoryPlanner
        self.trajectory_planner = SimpleTrajectoryPlanner
    
    def compute_grasp_poses(self, best_grasp):
        """
        Calculate pre-grasp and final grasp poses based on the best grasp
        
        Parameters:
            best_grasp: Best grasp pose (R, grasp_center)
            
        Returns:
            tuple: (pose1_pos, pose1_orn, pose2_pos, pose2_orn)
        """
        R, grasp_center = best_grasp
        
        # Build offset vector in gripper coordinate system
        local_offset = np.array([0, 0.06, 0])
        
        # Transform offset vector from gripper coordinate system to world coordinate system
        world_offset = R @ local_offset
        
        # Calculate compensated end-effector target position
        ee_target_pos = grasp_center + world_offset
        
        # Add coordinate system transformation
        combined_transform = np.array([
            [0, -1, 0],
            [0, 0, -1],
            [1, 0, 0]
        ])
        
        # Apply combined transformation
        R_world = R @ combined_transform
        
        # Convert rotation matrix to quaternion
        from scipy.spatial.transform import Rotation
        rot_world = Rotation.from_matrix(R_world)
        euler_world = rot_world.as_euler('xyz', degrees=True)
        
        # Define pose 2 (final grasp pose)
        pose2_pos = ee_target_pos
        pose2_orn = p.getQuaternionFromEuler([euler_world[0]/180*np.pi, euler_world[1]/180*np.pi, euler_world[2]/180*np.pi])
        
        # Calculate pose 1 (pre-grasp position) - move along z-axis of pose 2 backwards
        pose2_rot_matrix = R_world
        z_axis = pose2_rot_matrix[:, 2]
        pose1_pos = pose2_pos - 0.15 * z_axis
        pose1_orn = pose2_orn
        
        return pose1_pos, pose1_orn, pose2_pos, pose2_orn
    
    def execute_grasp(self, best_grasp):
        """
        Execute complete grasping process
        
        Parameters:
            best_grasp: Best grasp pose (R, grasp_center)
            
        Returns:
            bool: True if grasping is successful, False otherwise
        """
        # Calculate grasping pose
        pose1_pos, pose1_orn, pose2_pos, pose2_orn = self.compute_grasp_poses(best_grasp)
        
        # Get current robot joint angles
        start_joints = self.sim.robot.get_joint_positions()
        
        # Solve IK for pre-grasp position
        target_joints = self.ik_solver.solve(pose1_pos, pose1_orn, start_joints, max_iters=50, tolerance=0.001)
        
        if target_joints is None:
            print("IK cannot be solved, cannot move to pre-grasp position")
            return False
        
        # Generate and execute trajectory to pre-grasp position
        trajectory = self.trajectory_planner.generate_joint_trajectory(start_joints, target_joints, steps=100)
        self._execute_trajectory(trajectory)
        
        # Open gripper
        self.open_gripper()
        
        # Move to final grasp position
        current_joints = self.sim.robot.get_joint_positions()
        pose2_trajectory = self.trajectory_planner.generate_cartesian_trajectory(
            self.sim.robot.id, 
            self.sim.robot.arm_idx, 
            self.sim.robot.ee_idx,
            current_joints, 
            pose2_pos, 
            pose2_orn, 
            steps=50
        )
        
        if not pose2_trajectory:
            print("Cannot generate trajectory to final grasp position")
            return False
        
        self._execute_trajectory(pose2_trajectory)
        
        # Wait for stabilization
        self._wait(0.5)
        
        # Close gripper to grasp object
        self.close_gripper()
        
        # Lift object
        success = self.lift_object()
        
        return success
    
    def _execute_trajectory(self, trajectory, speed=1/240.0):
        """Execute trajectory"""
        for joint_target in trajectory:
            self.sim.robot.position_control(joint_target)
            for _ in range(1):
                self.sim.step()
                import time
                time.sleep(speed)
    
    def _wait(self, seconds):
        """Wait for specified seconds"""
        import time
        steps = int(seconds * 240)
        for _ in range(steps):
            self.sim.step()
            time.sleep(1/240.)
    
    def open_gripper(self, width=0.04):
        """Open robot gripper"""
        p.setJointMotorControlArray(
            self.sim.robot.id,
            jointIndices=self.sim.robot.gripper_idx,
            controlMode=p.POSITION_CONTROL,
            targetPositions=[width, width]
        )
        self._wait(0.5)
    
    def close_gripper(self, width=0.005):
        """Close robot gripper to grasp object"""
        p.setJointMotorControlArray(
            self.sim.robot.id,
            jointIndices=self.sim.robot.gripper_idx,
            controlMode=p.POSITION_CONTROL,
            targetPositions=[width, width]
        )
        self._wait(1.0)
    
    def lift_object(self, height=0.5):
        """Grasp object and lift it to specified height"""
        # Get current end-effector position and orientation
        current_ee_pos, current_ee_orn = self.sim.robot.get_ee_pose()
        
        # Calculate lifted position
        lift_pos = current_ee_pos.copy()
        lift_pos[2] += height
        
        # Get current joint angles
        current_joints = self.sim.robot.get_joint_positions()
        
        # Solve IK for lifted position
        lift_target_joints = self.ik_solver.solve(lift_pos, current_ee_orn, current_joints, max_iters=50, tolerance=0.001)
        
        if lift_target_joints is None:
            print("IK cannot be solved for lifted position, cannot lift object")
            return False
        
        # Generate and execute lifting trajectory
        lift_trajectory = self.trajectory_planner.generate_joint_trajectory(current_joints, lift_target_joints, steps=100)
        
        if not lift_trajectory:
            print("Cannot generate lifting trajectory")
            return False
        
        self._execute_trajectory(lift_trajectory, speed=1/240.0)
        return True

    def execute_complete_grasp(self, bbox, point_clouds, visualize=True):
        """
        Execute complete process of grasping planning and execution
        
        Parameters:
        bbox: Boundary box object
        point_clouds: Collected point cloud data
        visualize: Whether to visualize grasping process
        
        Returns:
        success: True if grasping is successful, False otherwise
        self: Grasping executor object (if grasping is successful)
        """
        import open3d as o3d
        from src.grasping import grasping_mesh
        from src.point_cloud.object_mesh import visualize_3d_objs
        
        print("\nStep 3: Grasping planning and execution...")
        
        # Merge point clouds
        print("\nPreparing to merge point clouds...")
        merged_pcd = None
        for data in point_clouds:
            if 'point_cloud' in data and data['point_cloud'] is not None:
                if merged_pcd is None:
                    merged_pcd = data['point_cloud']
                else:
                    merged_pcd += data['point_cloud']
        
        if merged_pcd is None:
            print("Error: Cannot merge point clouds, grasping terminated")
            return False, None
        
        # Get boundary box information
        center = bbox.get_center()
        rotation_matrix = bbox.get_rotation_matrix()
        min_point, max_point = bbox.get_aabb()
        obb_corners = bbox.get_corners()
        
        # Get rotated boundary box coordinates
        points_rotated = np.dot(np.asarray(merged_pcd.points) - center, rotation_matrix)
        min_point_rotated = np.min(points_rotated, axis=0)
        max_point_rotated = np.max(points_rotated, axis=0)
        
        print(f"\nBoundary box information:")
        print(f"Centroid coordinates: {center}")
        print(f"Minimum point in rotated coordinate system: {min_point_rotated}")
        print(f"Maximum point in rotated coordinate system: {max_point_rotated}")
        
        grasp_generator = GraspGeneration()
        
        # Generate grasping candidates
        print("\nGenerating grasping candidates...")
        sampled_grasps = grasp_generator.sample_grasps(
            center, 
            num_grasps=100, 
            sim=self.sim,
            rotation_matrix=rotation_matrix,
            min_point_rotated=min_point_rotated,
            max_point_rotated=max_point_rotated,
            center_rotated=center
        )
        
        # Create mesh for each grasping candidate
        all_grasp_meshes = []
        for grasp in sampled_grasps:
            R, grasp_center = grasp
            all_grasp_meshes.append(grasping_mesh.create_grasp_mesh(center_point=grasp_center, rotation_matrix=R))
        
        # Evaluate grasping quality
        print("\nEvaluating grasping quality...")
        
        best_grasp = None
        best_grasp_mesh = None
        highest_quality = 0
        
        for (pose, grasp_mesh) in zip(sampled_grasps, all_grasp_meshes):
            if not grasp_generator.check_grasp_collision(grasp_mesh, object_pcd=merged_pcd, num_colisions=1):
                R, grasp_center = pose
                
                valid_grasp, grasp_quality, _ = grasp_generator.check_grasp_containment(
                    grasp_mesh[0].get_center(), 
                    grasp_mesh[1].get_center(),
                    finger_length=0.05,
                    object_pcd=merged_pcd,
                    num_rays=50,
                    rotation_matrix=pose[0],
                    visualize_rays=False
                )
                
                if valid_grasp and grasp_quality > highest_quality:
                    highest_quality = grasp_quality
                    best_grasp = pose
                    best_grasp_mesh = grasp_mesh
                    print(f"Found better grasp, quality: {grasp_quality:.3f}")
        
        if best_grasp is None:
            print("No valid grasp found!")
            return False, None
        
        print(f"\nFound best grasp, quality score: {highest_quality:.4f}")
        
        # Calculate grasping pose
        pose1_pos, pose1_orn, pose2_pos, pose2_orn = self.compute_grasp_poses(best_grasp)
        
        # Visualize grasping pose
        if visualize:
            grasp_generator.visualize_grasp_poses(
                pose1_pos, pose1_orn, pose2_pos, pose2_orn, axis_length=0.1
            )
        
        # Execute grasping
        print("\nStarting to execute grasping...")
        success = self.execute_grasp(best_grasp)
        
        if success:
            print("\nGrasping successful!")
        else:
            print("\nGrasping failed...")
        
        # Add visualization code after finding the best grasp
        if best_grasp is not None and visualize:
            # Create triangle mesh from point cloud
            obj_triangle_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                pcd=merged_pcd, 
                alpha=0.08
            )
            
            # Prepare list of meshes for visualization
            vis_meshes = [obj_triangle_mesh]
            
            # Add best grasp mesh to list
            vis_meshes.extend(best_grasp_mesh)
            
            # Call visualization function
            visualize_3d_objs(vis_meshes)
        
        return success, self if success else None