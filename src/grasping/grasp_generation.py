import numpy as np
import open3d as o3d
import pybullet as p

from typing import Tuple, Sequence, Optional
from scipy.spatial.transform import Rotation
from src.grasping.mesh import visualize_3d_objs,create_grasp_mesh



class GraspGeneration:
    def __init__(self, bbox_center, bbox_rotation_matrix, sim):
        self.bbox_center = bbox_center
        self.bbox_rotation_matrix = bbox_rotation_matrix
        self.sim = sim

    def sample_grasps_state(
        self,
        center_point: np.ndarray,
        num_grasps: int,
        sim = None,
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
            rotation_matrix: OBB rotation matrix (from OBB coordinate system to world coordinate system)
            min_point_rotated: Minimum point of OBB in rotated coordinate system
            max_point_rotated: Maximum point of OBB in rotated coordinate system
            center_rotated: Origin of the OBB rotated coordinate system in world coordinates

        Returns:
            list: List of rotation matrices and translation vectors
        """
        grasp_poses_list = []
        table_height = sim.robot.pos[2] + 0.01
        
        grasp_points = []
        grasp_directions = []  # save the grasp direction
        
        obb_dims = max_point_rotated - min_point_rotated
        # height_threshold = 0.15  # 15 centimeters
        
        x_size = obb_dims[0]
        y_size = obb_dims[1]
        z_size = obb_dims[2]
              
        bbox_x_axis = rotation_matrix[:, 0]  # X axis of the Bounding box
        bbox_y_axis = rotation_matrix[:, 1]  # Y axis of the Bounding box
        
        # determine the short axis and long axis
        if x_size < y_size:
            short_axis = bbox_x_axis
            long_axis = bbox_y_axis
        else:
            short_axis = bbox_y_axis
            long_axis = bbox_x_axis
              
        # sample the position in the bounding box
        for idx in range(num_grasps):
            rotated_coords = np.zeros(3)
            rotated_coords[0] = np.random.uniform(min_point_rotated[0], max_point_rotated[0])
            rotated_coords[1] = np.random.uniform(min_point_rotated[1], max_point_rotated[1])
            rotated_coords[2] = np.random.uniform(min_point_rotated[2], max_point_rotated[2])
            
            # convert the sampled point from the rotated coordinate system to the world coordinate system
            grasp_center = np.dot(rotated_coords, rotation_matrix.T) + center_rotated
            
            # the grasp point is not lower than the table height
            grasp_center[2] = max(grasp_center[2], table_height)
            
             
            # Z axis is vertical downward
            grasp_z_axis = np.array([0, 0, -1])
            
            # X axis (the thickness direction of the gripper) uses the long axis
            grasp_x_axis = long_axis
            
            # Y axis (the opening direction of the gripper) uses the short axis
            grasp_y_axis = short_axis

            # ensure the coordinate system direction is correct
            if np.dot(np.cross(grasp_x_axis, grasp_y_axis), grasp_z_axis) < 0:
                grasp_y_axis = -grasp_y_axis
    
            # build the rotation matrix
            R = np.column_stack((grasp_x_axis, grasp_y_axis, grasp_z_axis))
            grasp_center = grasp_center - 0.05 * grasp_z_axis
            # # save the rotation matrix for visualization
            # grasp_directions.append(R)
            
            # add the grasp pose to the result list
            grasp_poses_list.append((R, grasp_center))
        
        # # define the length of the sampling point axis and color
        # axis_length = 0.05  # axis length
        # x_color = [1, 0, 0]  # x axis red
        # y_color = [0, 1, 0]  # y axis green
        # z_color = [0, 0, 1]  # z axis blue
        
        # for i, (point, direction) in enumerate(zip(grasp_points, grasp_directions)):
        #     # use the red sphere to represent the grasp point
        #     p.addUserDebugPoints([point], [[1, 0, 0]], pointSize=5, lifeTime=0)
            
        #     # extract the direction vectors of the three axes
        #     x_axis = direction[:, 0] * axis_length
        #     y_axis = direction[:, 1] * axis_length
        #     z_axis = direction[:, 2] * axis_length
            
        #     # draw the three axes
        #     p.addUserDebugLine(point, np.array(point) + x_axis, x_color, lineWidth=2, lifeTime=0)
        #     p.addUserDebugLine(point, np.array(point) + y_axis, y_color, lineWidth=2, lifeTime=0)
        #     p.addUserDebugLine(point, np.array(point) + z_axis, z_color, lineWidth=2, lifeTime=0)
            
        return grasp_poses_list
    
    def check_grasp_collision(
        self,
        grasp_meshes: Sequence[o3d.geometry.TriangleMesh],
        object_mesh: o3d.geometry.TriangleMesh = None,
        object_pcd = None,
        num_colisions: int = 10,
        tolerance: float = 0.00001) -> bool:

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

        # Calculate the height and bounding box of the object
        points = np.asarray(object_pcd.points)
        object_center = np.mean(points, axis=0)
        print(f"Object center point: {object_center}")

        obj_triangle_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd=object_pcd, 
                                                                                          alpha=0.016)
        
        obj_triangle_mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(obj_triangle_mesh)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(obj_triangle_mesh_t)

        hand_width = np.linalg.norm(left_center-right_center)
        ray_direction = (left_center - right_center)/hand_width
        finger_vec = np.array([0, 0, finger_length])

        # Store ray start and end points for visualization
        ray_start_points = []
        ray_end_points = []
        
        # ===== Calculate gripper width direction =====
        print("Calculating gripper width direction...")
        # Calculate vector in gripper width direction
        # First calculate finger_vec direction in world coordinates
        world_finger_vec = rotation_matrix.dot(finger_vec)
        # Calculate width direction vector
        width_direction = np.cross(ray_direction, world_finger_vec)
        width_direction = width_direction / np.linalg.norm(width_direction)
        
        # Define width direction parameters
        width_planes = 1  # Number of planes on each side in width direction
        width_offset = 0.01  # gripper thickness 0.02
        
        # ===== Generate multiple parallel ray planes =====
        print("Generating multiple parallel ray planes...")
        # Central plane (original plane)
        rays = []
        contained = False
        
        # Parallel planes on both sides in width direction
        for plane in range(1, width_planes + 1):
            # Calculate current plane offset
            current_offset = width_offset * plane
            
            # Right side plane
            for i in range(num_rays):
                # Calculate sampling point along length direction, and offset in width direction
                right_point = right_center - rotation_matrix.dot(0.5*finger_vec) + rotation_matrix.dot((i/num_rays)*finger_vec) + width_direction * current_offset
                # Add ray from right offset point to left offset point
                rays.append([np.concatenate([right_point, ray_direction])])
                
                # Store ray start and end points for visualization - using actual finger width
                ray_start_points.append(right_point)
                ray_end_points.append(right_point + ray_direction * hand_width)
            
            # Left side plane
            for i in range(num_rays):
                # Calculate sampling point along length direction, and offset in width direction
                right_point = right_center - rotation_matrix.dot(0.5*finger_vec) + rotation_matrix.dot((i/num_rays)*finger_vec) - width_direction * current_offset
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
                    lifeTime=0  # won't disappear
                )
                debug_lines.append(line_id)
        
        # Execute ray casting
        rays_t = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
        ans = scene.cast_rays(rays_t)
        
        # Process ray casting results
        rays_hit = 0
        max_interception_depth_score = o3d.core.Tensor([0.0], dtype=o3d.core.Dtype.Float32)
        center_rays_from_left = []
        center_rays_from_right = []
        # Track hits for left and right side ray planes
        left_side_hit = False
        right_side_hit = False
    
        # Process results for all rays
        print("Processing ray casting results...")

        for idx, hit_point in enumerate(ans['t_hit']):
            # Use actual finger width to determine if ray hit the object
            if hit_point < hand_width:
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
                    left_new_center = left_center - rotation_matrix.dot(0.5*finger_vec) + rotation_matrix.dot((idx/num_rays)*finger_vec)
                    right_new_center = right_center - rotation_matrix.dot(0.5*finger_vec) + rotation_matrix.dot((idx/num_rays)*finger_vec)
                    center_rays_from_left.append([np.concatenate([left_new_center, -ray_direction])])
                    center_rays_from_right.append([np.concatenate([right_new_center, ray_direction])])
        
        # Only consider contained when both left and right side planes have at least one ray hit
        contained = left_side_hit and right_side_hit
        
        containment_ratio = 0.0
        if contained:
            # Process rays from left side (only for center plane)
            if center_rays_from_left and center_rays_from_right:
                left_rays_t = o3d.core.Tensor(center_rays_from_left, dtype=o3d.core.Dtype.Float32)
                ans_left = scene.cast_rays(left_rays_t)
                right_rays_t = o3d.core.Tensor(center_rays_from_left, dtype=o3d.core.Dtype.Float32)
                ans_right = scene.cast_rays(right_rays_t)

                if(len(ans_left['t_hit']) == len(ans_right['t_hit'])):
                    hit_point_number = len(ans_left['t_hit'])
                    for idx in range(hit_point_number):
                        interception_depth = hand_width - ans_left['t_hit'][idx].item() - ans_right['t_hit'][idx].item()
                        max_interception_depth_score = max(max_interception_depth_score, interception_depth)


        print(f"the max interception depth is {max_interception_depth_score}")
        # Calculate overall ray hit ratio
        total_rays = len(rays)
        containment_ratio = rays_hit / total_rays
        print(f"Ray hit ratio: {containment_ratio:.4f} ({rays_hit}/{total_rays})")
        
        grasp_center = (left_center + right_center) / 2
        
        distance_to_center = np.linalg.norm(grasp_center - object_center)
        
        # Calculate distance score (closer distance gives higher score)
        center_score = np.exp(-distance_to_center**2 / (2 * 0.05**2))
      
        # Incorporate both distance scores into final quality score
        final_quality = 0.1 * containment_ratio + 0.1 * center_score + 80 * (1-np.exp(-max_interception_depth_score * 1000))
        
        print(f"Grasp center: {grasp_center}")
        print(f"Total distance: {distance_to_center}m, Total distance score: {center_score}")
        print(f"Final quality score: {final_quality}")
        
        return contained, final_quality

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
        
    def compute_grasp_poses(self, best_grasp):
        """
        Calculate pre-grasp and final grasp poses based on the best grasp
        
        Parameters:
            best_grasp: Best grasp pose (R, grasp_center)
            
        Returns:
            tuple: (pose1_pos, pose1_orn, pose2_pos, pose2_orn)
        """
        R, grasp_center = best_grasp
        
        ee_target_pos = grasp_center
        
        # Convert rotation matrix to quaternion
        rot_world = Rotation.from_matrix(R)
        euler_world = rot_world.as_euler('xyz', degrees=True)
        
        # Define pose 2 (final grasp pose)
        pose2_pos = ee_target_pos
        pose2_orn = p.getQuaternionFromEuler([euler_world[0]/180*np.pi, euler_world[1]/180*np.pi, euler_world[2]/180*np.pi])
        
        # Calculate pose 1 (pre-grasp position) - move along z-axis of pose 2 backwards
        pose2_rot_matrix = R
        z_axis = pose2_rot_matrix[:, 2]
        pose1_pos = pose2_pos - 0.15 * z_axis
        pose1_orn = pose2_orn
        
        return pose1_pos, pose1_orn, pose2_pos, pose2_orn
    
    def final_compute_poses(self, merged_point_clouds, visualize=True, object_name: Optional[str] = None):
        """
        Calculate pre-grasp and final grasp poses based on the best grasp
        
        Parameters:
            best_grasp: Best grasp pose (R, grasp_center)
        """
        print("\nStep 3: Grasping planning and execution...")
        
        # Merge point clouds
        print("\nPreparing to merge point clouds...")
        merged_pcd = None
        for data in merged_point_clouds:
            merged_pcd = data['point_cloud']

        
        if merged_pcd is None:
            print("Error: Cannot merge point clouds, grasping terminated")
            return False, None
        
        # Get boundary box information
        center = self.bbox_center
        rotation_matrix = self.bbox_rotation_matrix
        
        # Get rotated boundary box coordinates
        points_rotated = np.dot(np.asarray(merged_pcd.points) - center, rotation_matrix)
        min_point_rotated = np.min(points_rotated, axis=0)
        max_point_rotated = np.max(points_rotated, axis=0)
        
        print(f"\nBoundary box information:")
        print(f"Centroid coordinates: {center}")
        print(f"Minimum point in rotated coordinate system: {min_point_rotated}")
        print(f"Maximum point in rotated coordinate system: {max_point_rotated}")
        
        # Generate grasping candidates
        print("\nGenerating grasping candidates...")
        sampled_grasps_state = self.sample_grasps_state(
            center, 
            num_grasps=2000, 
            sim=self.sim,
            rotation_matrix=rotation_matrix,
            min_point_rotated=min_point_rotated,
            max_point_rotated=max_point_rotated,
            center_rotated=center
        )
        
        # Create mesh for each grasping candidate
        all_grasp_meshes = []
        
        for grasp in sampled_grasps_state:
            R, grasp_center = grasp
            gripper_meshes = create_grasp_mesh(center_point=grasp_center, rotation_matrix=R)
            all_grasp_meshes.append(gripper_meshes)
        # Visualize all grasp meshes
        print("\nVisualizing all grasp candidates...")
        # Create triangle mesh from point cloud for visualization
        obj_triangle_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            pcd=merged_pcd, 
            alpha=0.08
        )
        
        # # Prepare list of meshes for visualization
        # vis_meshes = [obj_triangle_mesh]
        
        # # Add all grasp meshes to list
        # for grasp_mesh in all_grasp_meshes:
        #     vis_meshes.extend(grasp_mesh)
            
        # # Call visualization function
        # visualize_3d_objs(vis_meshes)
        # # Evaluate grasping quality
        # print("\nEvaluating grasping quality...")
        
        best_grasp = None
        best_grasp_mesh = None
        highest_quality = 0

        if_collision = False

        for (pose, grasp_mesh) in zip(sampled_grasps_state, all_grasp_meshes):
            if object_name == "YcbPowerDrill":
                if_collision = self.check_grasp_collision(grasp_mesh, object_mesh=None, object_pcd = merged_pcd , num_colisions=1)
            else:
                if_collision = self.check_grasp_collision(grasp_mesh, object_mesh= obj_triangle_mesh, object_pcd = None , num_colisions=1)

            if not if_collision:
                R, grasp_center = pose
                
                valid_grasp, grasp_quality = self.check_grasp_containment(
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
                    print(f"Found better grasp, quality: {grasp_quality}")
        
        if best_grasp is None:
            print("No valid grasp found!")
            return False, None
        
        print(f"\nFound best grasp, quality score: {highest_quality}")
        
        # Calculate grasping pose (only calculate once)
        grasp_poses = self.compute_grasp_poses(best_grasp)
        pose1_pos, pose1_orn, pose2_pos, pose2_orn = grasp_poses

        # Visualize grasping pose
        if visualize:
            self.visualize_grasp_poses(
                pose1_pos, pose1_orn, pose2_pos, pose2_orn, axis_length=0.1
            )
              
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

        return pose1_pos, pose1_orn, pose2_pos, pose2_orn