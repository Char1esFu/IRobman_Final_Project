import numpy as np
import open3d as o3d
import pybullet as p

from typing import Tuple, Sequence
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
        grasp_directions = []  # 存储抓取方向（旋转矩阵）
        
        obb_dims = max_point_rotated - min_point_rotated
 
        
        x_size = obb_dims[0]
        y_size = obb_dims[1]
        z_size = obb_dims[2]
              
        bbox_x_axis = rotation_matrix[:, 0]  # Bounding box的X轴方向
        bbox_y_axis = rotation_matrix[:, 1]  # Bounding box的Y轴方向
        
        # 确定短边和长边对应的轴
        if x_size < y_size:
            short_axis = bbox_x_axis
            long_axis = bbox_y_axis
        else:
            short_axis = bbox_y_axis
            long_axis = bbox_x_axis
              
        # 在bounding box内均匀采样位置
        for idx in range(num_grasps):
            rotated_coords = np.zeros(3)
            rotated_coords[0] = np.random.uniform(min_point_rotated[0], max_point_rotated[0])
            rotated_coords[1] = np.random.uniform(min_point_rotated[1], max_point_rotated[1])
            rotated_coords[2] = np.random.uniform(min_point_rotated[2], max_point_rotated[2])
            
            # 将采样点从旋转坐标系转回世界坐标系
            grasp_center = np.dot(rotated_coords, rotation_matrix.T) + center_rotated
            
            # 抓取点不低于桌面高度
            grasp_center[2] = max(grasp_center[2], table_height)
            

            # Z轴垂直向下
            grasp_z_axis = np.array([0, 0, -1])
            
            # X轴（爪子厚度方向）使用长边
            grasp_x_axis = long_axis
            
            # Y轴（爪子开合方向）使用短边
            grasp_y_axis = short_axis


            # 确保坐标系方向正确
            if np.dot(np.cross(grasp_x_axis, grasp_y_axis), grasp_z_axis) < 0:
                grasp_y_axis = -grasp_y_axis
    
            # 构建旋转矩阵
            R = np.column_stack((grasp_x_axis, grasp_y_axis, grasp_z_axis))
            grasp_center = grasp_center - 0.05 * grasp_z_axis
            # # 保存旋转矩阵用于可视化
            # grasp_directions.append(R)
            
            # 将抓取姿态添加到结果列表
            grasp_poses_list.append((R, grasp_center))
        
        # # 定义采样点轴的长度和颜色
        # axis_length = 0.05  # 坐标轴长度，单位米
        # x_color = [1, 0, 0]  # 红色表示X轴（爪子厚度方向）
        # y_color = [0, 1, 0]  # 绿色表示Y轴（爪子开合方向）
        # z_color = [0, 0, 1]  # 蓝色表示Z轴（爪子朝向）
        
        # for i, (point, direction) in enumerate(zip(grasp_points, grasp_directions)):
        #     # 使用红色小球表示抓取点
        #     p.addUserDebugPoints([point], [[1, 0, 0]], pointSize=5, lifeTime=0)
            
        #     # 提取三个轴的方向向量
        #     x_axis = direction[:, 0] * axis_length
        #     y_axis = direction[:, 1] * axis_length
        #     z_axis = direction[:, 2] * axis_length
            
        #     # 画出三个坐标轴
        #     p.addUserDebugLine(point, np.array(point) + x_axis, x_color, lineWidth=2, lifeTime=0)
        #     p.addUserDebugLine(point, np.array(point) + y_axis, y_color, lineWidth=2, lifeTime=0)
        #     p.addUserDebugLine(point, np.array(point) + z_axis, z_color, lineWidth=2, lifeTime=0)
            
        return grasp_poses_list
    
    def check_grasp_collision(
        self,
        grasp_meshes: Sequence[o3d.geometry.TriangleMesh],
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

        if object_pcd is not None:
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
        width_offset = 0.01  # Offset between planes (meters)
        
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
    
    def final_compute_poses(self, merged_point_clouds, visualize=True):
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
            num_grasps=200, 
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
        
        # Prepare list of meshes for visualization
        vis_meshes = [obj_triangle_mesh]
        
        # Add all grasp meshes to list
        for grasp_mesh in all_grasp_meshes:
            vis_meshes.extend(grasp_mesh)
            
        # Call visualization function
        visualize_3d_objs(vis_meshes)
        # Evaluate grasping quality
        print("\nEvaluating grasping quality...")
        
        best_grasp = None
        best_grasp_mesh = None
        highest_quality = 0
        
        for (pose, grasp_mesh) in zip(sampled_grasps_state, all_grasp_meshes):
            if not self.check_grasp_collision(grasp_mesh, object_pcd=merged_pcd, num_colisions=1):
                R, grasp_center = pose
                
                valid_grasp, grasp_quality, _ = self.check_grasp_containment(
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