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
        sim = None,
        radius: float = 0.1,  # 保留参数但不使用
        rotation_matrix: np.ndarray = None,
        min_point_rotated: np.ndarray = None,
        max_point_rotated: np.ndarray = None,
        center_rotated: np.ndarray = None
    ) -> Sequence[Tuple[np.ndarray, np.ndarray]]:
        """
        在边界框内生成多个随机抓取姿态。

        参数:
            center_point: 点云的质心坐标
            num_grasps: 要生成的随机抓取姿态数量
            sim: 模拟对象
            radius: 最大距离偏移（保留但不使用）
            rotation_matrix: OBB旋转矩阵（从OBB坐标系到世界坐标系）
            min_point_rotated: OBB在旋转坐标系中的最小点
            max_point_rotated: OBB在旋转坐标系中的最大点
            center_rotated: OBB旋转坐标系的原点在世界坐标系中的位置

        返回:
            list: 旋转矩阵和平移向量的列表
        """
        if rotation_matrix is None or min_point_rotated is None or max_point_rotated is None or center_rotated is None:
            raise ValueError("必须提供旋转矩阵和OBB坐标")

        grasp_poses_list = []
        table_height = sim.robot.pos[2] # 机器人基座高度
        
        # 计算OBB三个维度的大小
        obb_dims = max_point_rotated - min_point_rotated
        
        # 确定维度的长短排序
        dims_with_idx = [(obb_dims[i], i) for i in range(3)]
        dims_with_idx.sort()  # 按照维度长度升序排序
        
        # 获取最短边索引和次短边索引
        shortest_axis_idx = dims_with_idx[0][1]  # 最短边对应的坐标轴索引
        second_shortest_idx = dims_with_idx[1][1]  # 次短边对应的坐标轴索引
        longest_axis_idx = dims_with_idx[2][1]  # 最长边对应的坐标轴索引

        # 判断物体是否足够高以应用完整的边界框对齐策略
        # 设置高度阈值（单位：米）
        height_threshold = 0.35  # 35厘米
        
        # 计算物体在世界坐标系中的垂直高度（z轴方向）
        object_z_height = obb_dims[2]  # 假设物体坐标系的第3个维度对应垂直方向
        
        # 判断垂直高度是否超过阈值
        is_tall_object = object_z_height > height_threshold

        # 如果是扁平物体，则预先计算质心在OBB坐标系中的坐标，用于后续采样
        if not is_tall_object:
            # 将质心从世界坐标系转换到OBB坐标系
            if center_point is not None:
                # center_point是世界坐标系中的质心，需要转换到OBB坐标系
                center_relative_to_obb = center_point - center_rotated
                center_in_obb = np.dot(center_relative_to_obb, rotation_matrix)
            else:
                # 如果未提供质心，使用OBB的中心
                center_in_obb = (min_point_rotated + max_point_rotated) / 2
        
        for idx in range(num_grasps):
            # 在OBB内采样，最短边使用正态分布
            rotated_coords = [0, 0, 0]
            
            # 对最短边使用正态分布采样，使中间位置密集
            min_val = min_point_rotated[shortest_axis_idx]
            max_val = max_point_rotated[shortest_axis_idx]
            
            # 均值为边长中点
            mean = (min_val + max_val) / 2
            # 标准差设置为边长的1/6，这样边长范围约为正负3个标准差，覆盖99.7%的正态分布
            std = (max_val - min_val) / 6
            
            # 使用截断正态分布确保值在边界内
            while True:
                sample = np.random.normal(mean, std)
                if min_val <= sample <= max_val:
                    rotated_coords[shortest_axis_idx] = sample
                    break
            
            # 判断是否为扁平物体（竖直向下抓取情况）
            if not is_tall_object:
                # 在正视图中，除了最短边外的另一条边（可能是最长边或次短边）
                # 确定哪一个轴是竖直轴
                vertical_axis_idx = None
                axes = [rotation_matrix[:, i] for i in range(3)]
                
                # 计算每个轴与世界坐标系Z轴的点积，找到最接近竖直方向的轴
                z_world = np.array([0, 0, 1])
                z_dots = [abs(np.dot(axis, z_world)) for axis in axes]
                vertical_axis_idx = np.argmax(z_dots)
                
                # 找到在水平面上的两个轴（除了竖直轴）
                horizontal_axes = [i for i in range(3) if i != vertical_axis_idx]
                
                # 现在我们有两个水平轴，其中一个是最短边
                # 另一个就是我们要找的"另一条边"
                other_axis_idx = horizontal_axes[0] if horizontal_axes[0] != shortest_axis_idx else horizontal_axes[1]
                
                # 对这个"另一条边"使用以质心为中心的正态分布采样
                min_other = min_point_rotated[other_axis_idx]
                max_other = max_point_rotated[other_axis_idx]
                
                # 使用质心在该轴上的投影作为均值
                mean_other = center_in_obb[other_axis_idx]
                # 如果质心投影超出了边界框范围，使用边界框中点作为均值
                if mean_other < min_other or mean_other > max_other:
                    mean_other = (min_other + max_other) / 2
                
                # 标准差设置为边长的1/6
                std_other = (max_other - min_other) / 6
                
                # 使用截断正态分布采样
                while True:
                    sample = np.random.normal(mean_other, std_other)
                    if min_other <= sample <= max_other:
                        rotated_coords[other_axis_idx] = sample
                        break
                
                # 竖直轴（Z轴）使用均匀分布或偏向顶部的分布
                # 由于是竖直向下抓取，我们可能更希望从物体顶部附近开始抓取
                min_z = min_point_rotated[vertical_axis_idx]
                max_z = max_point_rotated[vertical_axis_idx]
                # 偏向顶部的采样（使用Beta分布或其他偏向分布）
                # 这里简化为均匀分布，但更靠近顶部
                top_bias = 0.7  # 偏向顶部的程度，0.5为均匀，越大越靠近顶部
                z_sample = min_z + (max_z - min_z) * (1 - np.random.beta(1, top_bias))
                rotated_coords[vertical_axis_idx] = z_sample
            else:
                # 对于高物体，其余两轴仍使用均匀分布
                rotated_coords[second_shortest_idx] = np.random.uniform(
                    min_point_rotated[second_shortest_idx],
                    max_point_rotated[second_shortest_idx]
                )
                rotated_coords[longest_axis_idx] = np.random.uniform(
                    min_point_rotated[longest_axis_idx], 
                    max_point_rotated[longest_axis_idx]
                )
            
            grasp_center_rotated = np.array(rotated_coords)
            
            # 将采样点从旋转坐标系变换回世界坐标系
            grasp_center = np.dot(grasp_center_rotated, rotation_matrix.T) + center_rotated
            
            # 确保抓取点不低于桌面高度
            grasp_center[2] = max(grasp_center[2], table_height)
            print(f"grasp_center: {grasp_center}")

            # 基于OBB确定抓取姿态
            # 1. 计算OBB三个维度的大小
            obb_dims = max_point_rotated - min_point_rotated
            
            # 2. 确定维度的长短排序
            dims_with_idx = [(obb_dims[i], i) for i in range(3)]
            dims_with_idx.sort()  # 按照维度长度升序排序
            
            # 获取最短边索引和次短边索引
            shortest_axis_idx = dims_with_idx[0][1]  # 最短边对应的坐标轴索引
            second_shortest_idx = dims_with_idx[1][1]  # 次短边对应的坐标轴索引
            longest_axis_idx = dims_with_idx[2][1]  # 最长边对应的坐标轴索引
            
            # 3. 从rotation_matrix提取三个坐标轴方向
            axes = [rotation_matrix[:, i] for i in range(3)]  # OBB的三个轴方向
            
            # 判断物体是否足够高以应用完整的边界框对齐策略
            # 设置高度阈值（单位：米）
            height_threshold = 0.35  # 35厘米
            
            # 计算物体在世界坐标系中的垂直高度（z轴方向），而不是最长边
            # 物体的z轴高度定义为边界框在z方向的尺寸
            object_z_height = obb_dims[2]  # 假设物体坐标系的第3个维度对应垂直方向
            
            # 判断垂直高度是否超过阈值
            is_tall_object = object_z_height > height_threshold
            
            # 4. 构建新的旋转矩阵
            if is_tall_object:
                # 对于高物体，使用完整的边界框对齐策略
                # 爪子X轴（指尖打开方向）对应最短边
                grasp_x_axis = axes[shortest_axis_idx]
                # 爪子Y轴（指尖延伸方向）对应次短边
                grasp_y_axis = axes[second_shortest_idx]
                # 爪子Z轴通过叉乘确定
                grasp_z_axis = np.cross(grasp_x_axis, grasp_y_axis)
                
                # 确保坐标系是右手系
                dot_product = np.dot(grasp_z_axis, axes[longest_axis_idx])
                if dot_product < 0:
                    grasp_z_axis = -grasp_z_axis
                
                # 构建基于物体主轴的旋转矩阵
                R = np.column_stack((grasp_x_axis, grasp_y_axis, grasp_z_axis))
            else:
                # 对于扁平物体，使用简化策略
                # 将Z轴与世界坐标系的Z轴对齐，指向上方
                grasp_z_axis = np.array([0, 0, 1])  # 竖直向上
                
                # 尝试让X轴（指尖打开方向）从物体最短边获取方向
                # 但首先确保该方向不与Z轴平行
                candidate_x = axes[shortest_axis_idx]
                dot_xz = np.dot(candidate_x, grasp_z_axis)
                
                if abs(dot_xz) > 0.9:  # 最短边几乎垂直，改用次短边
                    candidate_x = axes[second_shortest_idx]
                    dot_xz = np.dot(candidate_x, grasp_z_axis)
                    
                    if abs(dot_xz) > 0.9:  # 次短边也几乎垂直，使用固定横向方向
                        grasp_x_axis = np.array([1, 0, 0])
                    else:
                        # 投影次短边到水平面作为X轴
                        grasp_x_axis = candidate_x - grasp_z_axis * dot_xz
                else:
                    # 投影最短边到水平面作为X轴
                    grasp_x_axis = candidate_x - grasp_z_axis * dot_xz
                
                # 确保X轴非零且归一化
                if np.linalg.norm(grasp_x_axis) < 1e-6:
                    grasp_x_axis = np.array([1, 0, 0])
                else:
                    grasp_x_axis = grasp_x_axis / np.linalg.norm(grasp_x_axis)
                
                # 通过叉乘计算Y轴（保证正交性）
                grasp_y_axis = np.cross(grasp_z_axis, grasp_x_axis)
                grasp_y_axis = grasp_y_axis / np.linalg.norm(grasp_y_axis)
                
                # 重新计算X轴以确保严格正交
                grasp_x_axis = np.cross(grasp_y_axis, grasp_z_axis)
                grasp_x_axis = grasp_x_axis / np.linalg.norm(grasp_x_axis)
                
                # 旋转爪子使Y轴沿负Z方向（指向下方）
                # 爪子坐标系：X-指尖打开方向，Y-手指延伸方向，Z-手掌方向
                # 创建旋转矩阵：[-x, -z, -y]，使手指向下
                R_adjust = np.array([
                    [1, 0, 0],
                    [0, 0, -1],
                    [0, 1, 0]
                ])
                
                # 构建我们的基础旋转矩阵 
                R_base = np.column_stack((grasp_x_axis, grasp_y_axis, grasp_z_axis))
                
                # 应用调整旋转
                R = R_base @ R_adjust
            
            # 打印抓取策略信息
            if idx == 0:  # 只打印一次
                if is_tall_object:
                    print(f"物体z轴高度({object_z_height:.3f}m)超过阈值({height_threshold:.3f}m)，使用完整边界框对齐策略")
                else:
                    print(f"物体z轴高度({object_z_height:.3f}m)低于阈值({height_threshold:.3f}m)，使用简化抓取策略（指尖沿最短边，从上方抓取）")

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
        检查抓取器姿态与目标物体之间是否存在碰撞，使用点云采样方法。

        参数:
            grasp_meshes: 表示抓取器组件的网格几何列表
            object_mesh: 目标物体的三角网格（可选）
            object_pcd: 目标物体的点云（可选）
            num_colisions: 判定为碰撞的点数阈值
            tolerance: 判定碰撞的距离阈值（米）

        返回:
            bool: 如果抓取器与物体之间检测到碰撞则为True，否则为False
        """
        # 合并抓取器网格
        combined_gripper = o3d.geometry.TriangleMesh()
        for mesh in grasp_meshes:
            combined_gripper += mesh

        # 从网格采样点
        num_points = 5000  # 对两个网格进行子采样的点数
        gripper_pcl = combined_gripper.sample_points_uniformly(number_of_points=num_points)
        
        # 确定使用哪个物体表示
        if object_mesh is not None:
            object_pcl = object_mesh.sample_points_uniformly(number_of_points=num_points)
        elif object_pcd is not None:
            object_pcl = object_pcd
        else:
            raise ValueError("必须提供object_mesh或object_pcd中的至少一个参数")

        # 为物体点构建KD树
        is_collision = False
        object_kd_tree = o3d.geometry.KDTreeFlann(object_pcl)
        collision_count = 0
        for point in gripper_pcl.points:
            [_, idx, distance] = object_kd_tree.search_knn_vector_3d(point, 1)
            if distance[0] < tolerance:
                collision_count += 1
                if collision_count >= num_colisions:
                    return True  # 检测到碰撞

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
        width_offset = 0.015  # 平面间的偏移量（米）
        
        # ===== 生成多个平行的射线平面 =====
        print("生成多个平行的射线平面...")
        # 中心平面（原始平面）
        rays = []
        contained = False
        rays_hit = 0
        
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
                ray_end_points.append(right_point + ray_direction * hand_width)
            
            # 左侧平面
            for i in range(num_rays):
                # 计算长度方向上的采样点，并在宽度方向上偏移
                right_point = right_center + rotation_matrix.dot((i/num_rays)*finger_vec) - width_direction * current_offset
                # 添加从右侧偏移点到左侧偏移点的射线
                rays.append([np.concatenate([right_point, ray_direction])])
                
                # 存储射线起点和终点用于可视化 - 使用实际手指宽度
                ray_start_points.append(right_point)
                ray_end_points.append(right_point + ray_direction * hand_width)
        
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
        
        # 跟踪左侧和右侧射线平面的命中情况
        left_side_hit = False
        right_side_hit = False
        
        # 计算每个平面的射线数量
        rays_per_plane = num_rays
        
        # 处理所有射线的结果
        print("处理射线投射结果...")
        for idx, hit_point in enumerate(ans['t_hit']):
            # 使用实际手指宽度判断射线是否击中物体
            if hit_point[0] < hand_width:
                rays_hit += 1
                
                # 确定射线属于左侧还是右侧平面
                total_rays_count = len(rays)
                half_rays_count = total_rays_count // 2
                
                if idx < half_rays_count:
                    # 右侧平面的射线
                    right_side_hit = True
                else:
                    # 左侧平面的射线
                    left_side_hit = True
                
                # 只为中心平面（原始平面）的射线计算深度
                if idx < num_rays:
                    left_new_center = left_center + rotation_matrix.dot((idx/num_rays)*finger_vec)
                    rays_from_left.append([np.concatenate([left_new_center, -ray_direction])])
        
        # 只有当左侧和右侧平面都至少有一条射线命中时，才算作contained
        contained = left_side_hit and right_side_hit
        
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
                        if hitpoint[0] < hand_width: 
                            interception_depth = hand_width - ans_left['t_hit'][0].item() - hitpoint[0].item()
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

        # 计算抓取中心到物体中心的距离
        grasp_center = (left_center + right_center) / 2
        
        # 计算3D空间中的总距离
        distance_to_center = np.linalg.norm(grasp_center - object_center)
        
        # 计算仅在x-y平面上的距离（水平距离）
        horizontal_distance = np.linalg.norm(grasp_center[:2] - object_center[:2])
        
        # 计算距离分数（距离越近分数越高）
        center_score = np.exp(-distance_to_center**2 / (2 * 0.05**2))
        
        # 计算水平距离分数（水平距离越近分数越高）
        horizontal_score = np.exp(-horizontal_distance**2 / (2 * 0.03**2))
        
        # 将两个距离分数纳入最终质量评分，给水平距离更高的权重
        final_quality = containment_ratio * (1 + center_score + 1.5 * horizontal_score)
        
        print(f"抓取中心: {grasp_center}")
        print(f"水平距离: {horizontal_distance:.4f}m, 水平分数: {horizontal_score:.4f}")
        print(f"总距离: {distance_to_center:.4f}m, 总距离分数: {center_score:.4f}")
        print(f"最终质量评分: {final_quality:.4f}")
        
        return any(intersections), final_quality, max_interception_depth.item()

    def visualize_grasp_poses(self, 
                             pose1_pos, 
                             pose1_orn, 
                             pose2_pos, 
                             pose2_orn, 
                             axis_length=0.1):
        """
        在PyBullet中可视化抓取位姿坐标轴
        
        参数:
            pose1_pos: 预抓取位置
            pose1_orn: 预抓取方向（四元数）
            pose2_pos: 最终抓取位置
            pose2_orn: 最终抓取方向（四元数）
            axis_length: 坐标轴长度
        """
        # 从四元数获取旋转矩阵
        pose1_rot = np.array(p.getMatrixFromQuaternion(pose1_orn)).reshape(3, 3)
        pose2_rot = np.array(p.getMatrixFromQuaternion(pose2_orn)).reshape(3, 3)
        
        # 提取各个轴的方向向量
        pose1_x_axis = pose1_rot[:, 0] * axis_length
        pose1_y_axis = pose1_rot[:, 1] * axis_length
        pose1_z_axis = pose1_rot[:, 2] * axis_length
        
        pose2_x_axis = pose2_rot[:, 0] * axis_length
        pose2_y_axis = pose2_rot[:, 1] * axis_length
        pose2_z_axis = pose2_rot[:, 2] * axis_length
        
        # 可视化Pose 1的坐标轴
        p.addUserDebugLine(pose1_pos, pose1_pos + pose1_x_axis, [1, 0, 0], 3, 0)  # X轴 - 红色
        p.addUserDebugLine(pose1_pos, pose1_pos + pose1_y_axis, [0, 1, 0], 3, 0)  # Y轴 - 绿色
        p.addUserDebugLine(pose1_pos, pose1_pos + pose1_z_axis, [0, 0, 1], 3, 0)  # Z轴 - 蓝色
        
        # 可视化Pose 2的坐标轴
        p.addUserDebugLine(pose2_pos, pose2_pos + pose2_x_axis, [1, 0, 0], 3, 0)  # X轴 - 红色
        p.addUserDebugLine(pose2_pos, pose2_pos + pose2_y_axis, [0, 1, 0], 3, 0)  # Y轴 - 绿色
        p.addUserDebugLine(pose2_pos, pose2_pos + pose2_z_axis, [0, 0, 1], 3, 0)  # Z轴 - 蓝色
        
        # 添加文本标签
        p.addUserDebugText("Pose 1", pose1_pos + [0, 0, 0.05], [1, 1, 1], 1.5)
        p.addUserDebugText("Pose 2", pose2_pos + [0, 0, 0.05], [1, 1, 1], 1.5)

class GraspExecution:
    """机器人抓取执行类，负责规划和执行完整的抓取动作"""
    
    def __init__(self, sim):
        """
        初始化抓取执行器
        
        参数:
            sim: 模拟环境对象
        """
        self.sim = sim
        from src.ik_solver import DifferentialIKSolver
        self.ik_solver = DifferentialIKSolver(sim.robot.id, sim.robot.ee_idx, damping=0.05)
        from src.path_planning.planning import TrajectoryPlanner
        self.trajectory_planner = TrajectoryPlanner
    
    def compute_grasp_poses(self, best_grasp):
        """
        根据最佳抓取计算预抓取位姿和最终抓取位姿
        
        参数:
            best_grasp: 最佳抓取姿态(R, grasp_center)
            
        返回:
            tuple: (pose1_pos, pose1_orn, pose2_pos, pose2_orn)
        """
        R, grasp_center = best_grasp
        
        # 构建爪子自身坐标系中的偏移向量
        local_offset = np.array([0, 0.06, 0])
        
        # 使用旋转矩阵将偏移向量从爪子坐标系转换到世界坐标系
        world_offset = R @ local_offset
        
        # 计算补偿后的末端执行器目标位置
        ee_target_pos = grasp_center + world_offset
        
        # 添加坐标系转换
        combined_transform = np.array([
            [0, -1, 0],
            [0, 0, -1],
            [1, 0, 0]
        ])
        
        # 应用合并后的转换
        R_world = R @ combined_transform
        
        # 将旋转矩阵转换为四元数
        from scipy.spatial.transform import Rotation
        rot_world = Rotation.from_matrix(R_world)
        euler_world = rot_world.as_euler('xyz', degrees=True)
        
        # 定义pose 2（最终抓取位姿）
        pose2_pos = ee_target_pos
        pose2_orn = p.getQuaternionFromEuler([euler_world[0]/180*np.pi, euler_world[1]/180*np.pi, euler_world[2]/180*np.pi])
        
        # 计算pose 1（抓取前位置）- 沿着pose 2自身的z轴往后退
        pose2_rot_matrix = R_world
        z_axis = pose2_rot_matrix[:, 2]
        pose1_pos = pose2_pos - 0.15 * z_axis
        pose1_orn = pose2_orn
        
        return pose1_pos, pose1_orn, pose2_pos, pose2_orn
    
    def execute_grasp(self, best_grasp):
        """
        执行完整的抓取过程
        
        参数:
            best_grasp: 最佳抓取姿态(R, grasp_center)
            
        返回:
            bool: 抓取是否成功
        """
        # 计算抓取姿态
        pose1_pos, pose1_orn, pose2_pos, pose2_orn = self.compute_grasp_poses(best_grasp)
        
        # 获取当前机械臂关节角度
        start_joints = self.sim.robot.get_joint_positions()
        
        # 解算预抓取位置的IK
        target_joints = self.ik_solver.solve(pose1_pos, pose1_orn, start_joints, max_iters=50, tolerance=0.001)
        
        if target_joints is None:
            print("无法解算IK，无法移动到抓取前位置")
            return False
        
        # 生成并执行到预抓取位置的轨迹
        trajectory = self.trajectory_planner.generate_joint_trajectory(start_joints, target_joints, steps=100)
        self._execute_trajectory(trajectory)
        
        # 打开爪子
        self.open_gripper()
        
        # 移动到最终抓取位置
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
            print("无法生成到最终抓取位置的轨迹")
            return False
        
        self._execute_trajectory(pose2_trajectory)
        
        # 等待稳定
        self._wait(0.5)
        
        # 关闭爪子抓取物体
        self.close_gripper()
        
        # 提升物体
        success = self.lift_object()
        
        return success
    
    def _execute_trajectory(self, trajectory, speed=1/240.0):
        """执行轨迹"""
        for joint_target in trajectory:
            self.sim.robot.position_control(joint_target)
            for _ in range(1):
                self.sim.step()
                import time
                time.sleep(speed)
    
    def _wait(self, seconds):
        """等待指定的秒数"""
        import time
        steps = int(seconds * 240)
        for _ in range(steps):
            self.sim.step()
            time.sleep(1/240.)
    
    def open_gripper(self, width=0.04):
        """打开机器人爪子"""
        p.setJointMotorControlArray(
            self.sim.robot.id,
            jointIndices=self.sim.robot.gripper_idx,
            controlMode=p.POSITION_CONTROL,
            targetPositions=[width, width]
        )
        self._wait(0.5)
    
    def close_gripper(self, width=0.005):
        """关闭机器人爪子抓取物体"""
        p.setJointMotorControlArray(
            self.sim.robot.id,
            jointIndices=self.sim.robot.gripper_idx,
            controlMode=p.POSITION_CONTROL,
            targetPositions=[width, width]
        )
        self._wait(1.0)
    
    def lift_object(self, height=0.5):
        """抓取物体后将其提升到指定高度"""
        # 获取当前末端执行器位置和方向
        current_ee_pos, current_ee_orn = self.sim.robot.get_ee_pose()
        
        # 计算向上提升后的位置
        lift_pos = current_ee_pos.copy()
        lift_pos[2] += height
        
        # 获取当前关节角度
        current_joints = self.sim.robot.get_joint_positions()
        
        # 解算提升位置的IK
        lift_target_joints = self.ik_solver.solve(lift_pos, current_ee_orn, current_joints, max_iters=50, tolerance=0.001)
        
        if lift_target_joints is None:
            print("无法解算提升位置的IK，无法提升物体")
            return False
        
        # 生成并执行提升轨迹
        lift_trajectory = self.trajectory_planner.generate_joint_trajectory(current_joints, lift_target_joints, steps=100)
        
        if not lift_trajectory:
            print("无法生成提升轨迹")
            return False
        
        self._execute_trajectory(lift_trajectory, speed=1/240.0)
        return True