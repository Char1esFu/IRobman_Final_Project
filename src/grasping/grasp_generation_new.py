import numpy as np
from typing import Tuple, Sequence, Optional, Any
import open3d as o3d
import pybullet as p  # 导入pybullet用于可视化


class GraspGeneration:
    def __init__(self):
        pass

    def sample_grasps(
        self,
        center_point: np.ndarray,
        num_grasps: int,
        radius: float = 0.1,
    ) -> Sequence[Tuple[np.ndarray, np.ndarray]]:
        """
        Generates multiple random grasp poses around a given point cloud.

        Args:
            center: Center around which to sample grasps.
            num_grasps: Number of random grasp poses to generate
            radius: Maximum distance offset from the center (meters)

        Returns:
            list: List of rotations and Translations
        """

        grasp_poses_list = []
        for idx in range(num_grasps):
            # Sample a grasp center and rotation of the grasp
            # Sample a random vector in R3 for axis angle representation
            # Return the rotation as rotation matrix + translation
            # Translation implies translation from a center point
            theta = np.random.uniform(0, 2*np.pi)

            # phi = np.random.uniform(0, np.pi)
            # this creates a lot of points around the pole. The points are not uniformly distributed around the sphere.
            # There is some transformation that can be applied to the random variable to remedy this issue, TODO look into that

            # phi = np.arccos(1 - 2 * np.random.uniform(0, 1))
            phi = np.arccos(np.random.uniform(0, 1))
            # source https://math.stackexchange.com/questions/1585975/how-to-generate-random-points-on-a-sphere
            r = np.random.uniform(0, radius)

            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)
            grasp_center = center_point + np.array([x, y, z])

            # axis = np.random.normal(size=3)
            # axis = np.array([0, 0, -1])
            # axis /= np.linalg.norm(axis)
            # angle = np.random.uniform(0, 2 * np.pi)

            # K =  np.array([
            #     [0, -axis[2], axis[1]],
            #     [axis[2], 0, -axis[0]],
            #     [-axis[1], axis[0], 0],
            # ])
            # R = np.eye(3) + np.sin(angle)*K + (1 - np.cos(angle))*K.dot(K)

            # offset = np.random.uniform(0, np.pi/12)
            offset = 0
            
            Rx = np.array([
                [1,  0,  0],
                [ 0, np.cos(offset+np.pi/2),  -np.sin(offset+np.pi/2)],
                [ 0, np.sin(offset+np.pi/2),  np.cos(offset+np.pi/2)]
            ])
            
            # Generate a random angle for X rotation
            theta = np.random.uniform(0, 2 * np.pi)  # Random angle in radians
            cos_t, sin_t = np.cos(theta), np.sin(theta)

            # Rotation matrix about X-axis
            Ry = np.array([
                [cos_t, 0, sin_t],
                [ 0, 1, 0],
                [-sin_t, 0, cos_t]
            ])

            # Ry = np.eye(3)

            Rx_again = np.array([
                [1, 0, 0],
                [0, np.cos(offset), -np.sin(offset)],
                [0, np.sin(offset), np.cos(offset)]
            ])

            # Final rotation matrix: First apply Rx, then Rz
            R = Rx @ Ry @ Rx_again # Equivalent to R = np.dot(Rz, Rx)



            # assert R.shape == (3, 3)
            assert grasp_center.shape == (3,)
            grasp_poses_list.append((R, grasp_center))

        return grasp_poses_list
    

    def check_grasp_collision(
        self,
        grasp_meshes: Sequence[o3d.geometry.TriangleMesh],
        # object_mesh: o3d.geometry.TriangleMesh,
        object_pcd,
        num_colisions: int = 10,
        tolerance: float = 0.00001) -> bool:
        """
        Checks for collisions between a gripper grasp pose and target object
        using point cloud sampling.

        Args:
            grasp_meshes: List of mesh geometries representing the gripper components
            object_mesh: Triangle mesh of the target object
            num_collisions: Threshold on how many points to check
            tolerance: Distance threshold for considering a collision (in meters)

        Returns:
            bool: True if collision detected between gripper and object, False otherwise
        """
        # Combine gripper meshes
        combined_gripper = o3d.geometry.TriangleMesh()
        for mesh in grasp_meshes:
            combined_gripper += mesh

        # Sample points from both meshes
        num_points = 5000 # Subsample both meshes to this many points
        gripper_pcl = combined_gripper.sample_points_uniformly(number_of_points=num_points)
        # object_pcl = object_mesh.sample_points_uniformly(number_of_points=num_points)
        object_pcl = object_pcd

        # Build KDTree for object points
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
        visualize_rays: bool = False  # 是否在PyBullet中可视化射线
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

        # 计算物体的高度和边界框
        points = np.asarray(object_pcd.points)
        min_point = np.min(points, axis=0)
        max_point = np.max(points, axis=0)
        object_height = max_point[2] - min_point[2]
        object_center = (min_point + max_point) / 2
        
        print(f"物体高度: {object_height:.4f}m")
        print(f"物体中心点: {object_center}")

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
        
        # 调整左右手指中心点，使其位于手指的实际位置
        # 将右手指中心点移动到手指的起始位置
        right_center = right_center - rotation_matrix.dot(finger_vec/2)
        # 同样调整左手指中心点
        left_center = left_center - rotation_matrix.dot(finger_vec/2)
        
        # 对于低高度物体，调整射线起点高度
        if object_height < 0.05:  # 如果物体高度小于5cm
            print("检测到低高度物体，调整射线高度...")
            
            # 计算手指中心点与物体中心点在z轴上的差距
            z_diff = (right_center[2] + left_center[2]) / 2 - object_center[2]
            
            # 如果手指中心点高于物体中心点，则将手指中心点降低到物体中心高度
            if z_diff > 0.01:  # 如果差距大于1cm
                height_adjustment = z_diff - 0.01  # 保留1cm的余量
                right_center[2] -= height_adjustment
                left_center[2] -= height_adjustment
                print(f"射线高度已调整: {height_adjustment:.4f}m")
        
        # 重新计算手指之间的实际距离
        actual_hand_width = np.linalg.norm(left_center-right_center)
        
        # 用于存储射线的起点和终点，用于可视化
        ray_start_points = []
        ray_end_points = []
        
        # ===== 计算爪子宽度方向 =====
        print("计算爪子宽度方向...")
        # 计算爪子宽度方向的向量（与ray_direction和finger_vec都垂直）
        # 首先计算finger_vec在世界坐标系中的方向
        world_finger_vec = rotation_matrix.dot(finger_vec)
        # 计算宽度方向向量（叉乘得到垂直于两个向量的第三个向量）
        width_direction = np.cross(ray_direction, world_finger_vec)
        # 归一化
        width_direction = width_direction / np.linalg.norm(width_direction)
        
        # 定义宽度方向的参数
        width_planes = 1  # 宽度方向每侧的平面数量
        width_offset = 0.02  # 平面间的偏移量（米）
        
        # ===== 生成多个平行的射线平面 =====
        print("生成多个平行的射线平面...")
        # 中心平面（原始平面）
        rays = []
        contained = False
        rays_hit = 0
        
        # 对于低高度物体，增加垂直方向的射线密度
        vertical_rays = 0
        if object_height < 0.05:  # 如果物体高度小于5cm
            print("为低高度物体增加垂直方向射线...")
            vertical_rays = 10  # 增加10条垂直方向的射线
            vertical_step = object_height / (vertical_rays + 1)  # 垂直方向的步长
        
        for i in range(num_rays):
            # 计算长度方向上的采样点
            right_new_center = right_center + rotation_matrix.dot((i/num_rays)*finger_vec)
            left_new_center = left_center + rotation_matrix.dot((i/num_rays)*finger_vec)
            # 添加从右指尖到左指尖的射线
            rays.append([np.concatenate([right_new_center, ray_direction])])
            
            # 存储射线起点和终点用于可视化 - 使用实际手指宽度
            ray_start_points.append(right_new_center)
            ray_end_points.append(right_new_center + ray_direction * actual_hand_width)
            
            # 为低高度物体添加垂直方向的射线
            if vertical_rays > 0:
                for v in range(vertical_rays):
                    # 计算垂直偏移
                    v_offset = (v + 1) * vertical_step
                    
                    # 向下的射线
                    down_point = right_new_center.copy()
                    down_point[2] -= v_offset
                    rays.append([np.concatenate([down_point, ray_direction])])
                    
                    # 存储射线起点和终点用于可视化
                    ray_start_points.append(down_point)
                    ray_end_points.append(down_point + ray_direction * actual_hand_width)
                    
                    # 向上的射线
                    up_point = right_new_center.copy()
                    up_point[2] += v_offset
                    rays.append([np.concatenate([up_point, ray_direction])])
                    
                    # 存储射线起点和终点用于可视化
                    ray_start_points.append(up_point)
                    ray_end_points.append(up_point + ray_direction * actual_hand_width)
        
        # 宽度方向两侧的平行平面
        for plane in range(1, width_planes + 1):
            # 计算当前平面的偏移量
            current_offset = width_offset * plane
            
            # 右侧平面
            for i in range(num_rays):
                # 计算长度方向上的采样点，并在宽度方向上偏移
                right_point = right_center + rotation_matrix.dot((i/num_rays)*finger_vec) + width_direction * current_offset
                # 添加从右侧偏移点到左侧偏移点的射线
                rays.append([np.concatenate([right_point, ray_direction])])
                
                # 存储射线起点和终点用于可视化 - 使用实际手指宽度
                ray_start_points.append(right_point)
                ray_end_points.append(right_point + ray_direction * actual_hand_width)
            
            # 左侧平面
            for i in range(num_rays):
                # 计算长度方向上的采样点，并在宽度方向上偏移
                right_point = right_center + rotation_matrix.dot((i/num_rays)*finger_vec) - width_direction * current_offset
                # 添加从右侧偏移点到左侧偏移点的射线
                rays.append([np.concatenate([right_point, ray_direction])])
                
                # 存储射线起点和终点用于可视化 - 使用实际手指宽度
                ray_start_points.append(right_point)
                ray_end_points.append(right_point + ray_direction * actual_hand_width)
        
        print(f"总共生成了 {len(rays)} 条射线")
        
        # 在PyBullet中可视化射线
        debug_lines = []
        if visualize_rays:
            print("在PyBullet中可视化射线...")
            for start, end in zip(ray_start_points, ray_end_points):
                line_id = p.addUserDebugLine(
                    start.tolist(), 
                    end.tolist(), 
                    lineColorRGB=[1, 0, 0],  # 红色
                    lineWidth=1,
                    lifeTime=5  # 5秒后自动消失
                )
                debug_lines.append(line_id)
        
        # 执行射线投射
        rays_t = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
        ans = scene.cast_rays(rays_t)
        
        # 处理射线投射结果
        rays_hit = 0
        max_interception_depth = o3d.core.Tensor([0.0], dtype=o3d.core.Dtype.Float32)
        rays_from_left = []
        
        # 处理所有射线的结果
        print("处理射线投射结果...")
        for idx, hit_point in enumerate(ans['t_hit']):
            # 使用实际手指宽度判断射线是否击中物体
            if hit_point[0] < actual_hand_width:
                contained = True
                rays_hit += 1
                
                # 只为中心平面（原始平面）的射线计算深度
                if idx < num_rays:
                    left_new_center = left_center + rotation_matrix.dot((idx/num_rays)*finger_vec)
                    rays_from_left.append([np.concatenate([left_new_center, -ray_direction])])
        
        containment_ratio = 0.0
        if contained:
            # 处理从左侧发出的射线（仅针对中心平面）
            if rays_from_left:
                rays_t = o3d.core.Tensor(rays_from_left, dtype=o3d.core.Dtype.Float32)
                ans_left = scene.cast_rays(rays_t)
                
                for idx, hitpoint in enumerate(ans['t_hit']):
                    if idx < num_rays:  # 只处理中心平面的射线
                        left_idx = 0
                        # 使用实际手指宽度计算截断深度
                        if hitpoint[0] < actual_hand_width: 
                            interception_depth = actual_hand_width - ans_left['t_hit'][0].item() - hitpoint[0].item()
                            max_interception_depth = max(max_interception_depth, interception_depth)
                            left_idx += 1

        print(f"the max interception depth is {max_interception_depth}")
        # 计算总的射线命中率
        total_rays = len(rays)
        containment_ratio = rays_hit / total_rays
        print(f"射线命中率: {containment_ratio:.4f} ({rays_hit}/{total_rays})")
        
        intersections.append(contained)
        # intersections.append(max_interception_depth[0])
        # return contained, containment_ratio

        return any(intersections), containment_ratio, max_interception_depth.item()