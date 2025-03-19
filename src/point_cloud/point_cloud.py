import os
import glob
import numpy as np
import pybullet as p
import open3d as o3d
import time
import random
from scipy.spatial.transform import Rotation
from src.path_planning.planning import TrajectoryPlanner

class PointCloudCollector:
    def __init__(self, config, sim):
        """
        初始化点云收集器
        
        参数:
        config: 配置字典
        sim: 模拟环境对象
        """
        self.config = config
        self.sim = sim
        
    def _convert_depth_to_meters(self, depth_buffer, near, far):
        """
        转换深度缓冲区值为实际距离（米）
        
        参数:
        depth_buffer: 从PyBullet获取的深度缓冲区值
        near, far: 近/远平面距离
        
        返回:
        以米为单位的实际深度值
        """
        return far * near / (far - (far - near) * depth_buffer)

    def _get_camera_intrinsic(self, width, height, fov):
        """
        从相机参数计算内参矩阵
        
        参数:
        width: 图像宽度（像素）
        height: 图像高度（像素）
        fov: 垂直视场角（度）

        返回:
        相机内参矩阵
        """    
        # 计算焦距
        f = height / (2 * np.tan(np.radians(fov / 2)))
        
        # 计算主点
        cx = width / 2
        cy = height / 2
        
        intrinsic_matrix = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1]
        ])
        
        return intrinsic_matrix

    def _depth_image_to_point_cloud(self, depth_image, mask, rgb_image, intrinsic_matrix):
        """
        深度图像转换为相机坐标系点云
        
        参数:
        depth_image: 深度图像（米）
        mask: 目标物体掩码（布尔数组）
        rgb_image: RGB图像
        intrinsic_matrix: 相机内参矩阵
        
        返回:
        相机坐标系点云(N,3)和对应的颜色(N,3)
        """
        # 提取目标掩码的像素坐标
        rows, cols = np.where(mask)
        
        if len(rows) == 0:
            raise ValueError("目标掩码中未找到有效像素")
        
        # 提取这些像素的深度值
        depths = depth_image[rows, cols]
        
        # 图像坐标转相机坐标
        fx = intrinsic_matrix[0, 0]
        fy = intrinsic_matrix[1, 1]
        cx = intrinsic_matrix[0, 2]
        cy = intrinsic_matrix[1, 2]
        
        # 计算相机坐标
        x = -(cols - cx) * depths / fx # 负号是由于PyBullet相机方向
        y = -(rows - cy) * depths / fy
        z = depths
        
        # 堆叠点
        points = np.vstack((x, y, z)).T
        
        # 提取RGB颜色
        colors = rgb_image[rows, cols, :3].astype(np.float64) / 255.0
        
        return points, colors

    def _transform_points_to_world(self, points, camera_extrinsic):
        """
        将点从相机坐标系转换到世界坐标系
        
        参数:
        points: 相机坐标系中的点云(N,3)
        camera_extrinsic: 相机外参矩阵(4x4)
        
        返回:
        世界坐标系中的点云(N,3)
        """
        # 转换点云为齐次坐标
        points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
        
        # 使用外参矩阵转换点云
        world_points_homogeneous = (camera_extrinsic @ points_homogeneous.T).T # 点在行中
        
        # 转换回非齐次坐标
        world_points = world_points_homogeneous[:, :3]
        
        return world_points

    def _get_camera_extrinsic(self, camera_pos, camera_R):
        """
        构建相机外参矩阵（从相机到世界坐标系的转换）
        
        参数:
        camera_pos: 世界坐标系中的相机位置
        camera_R: 相机旋转矩阵(3x3)
        
        返回:
        相机外参矩阵(4x4)
        """
        # 构建4x4外参矩阵
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = camera_R
        extrinsic[:3, 3] = camera_pos
        
        return extrinsic

    def _build_object_point_cloud_ee(self, rgb, depth, seg, target_mask_id, camera_pos, camera_R):
        """
        使用末端执行器相机的RGB、深度、分割数据构建物体点云
        
        参数:
        rgb: RGB图像
        depth: 深度缓冲区值
        seg: 分割掩码
        target_mask_id: 目标物体ID
        camera_pos: 世界坐标系中的相机位置
        camera_R: 相机旋转矩阵（从相机到世界坐标系）
        
        返回:
        Open3D点云对象
        """
        # 读取相机参数
        cam_cfg = self.config["world_settings"]["camera"]
        width = cam_cfg["width"]
        height = cam_cfg["height"]
        fov = cam_cfg["fov"]  # 垂直FOV
        near = cam_cfg["near"]
        far = cam_cfg["far"]
        
        # 创建目标物体掩码
        object_mask = (seg == target_mask_id)
        if np.count_nonzero(object_mask) == 0:
            raise ValueError(f"分割中未找到目标掩码ID {target_mask_id}")
        
        # 提取目标物体的深度缓冲区值
        metric_depth = self._convert_depth_to_meters(depth, near, far)
        
        # 获取内参矩阵
        intrinsic_matrix = self._get_camera_intrinsic(width, height, fov)
        
        # 将深度图像转换为点云
        points_cam, colors = self._depth_image_to_point_cloud(metric_depth, object_mask, rgb, intrinsic_matrix)
        
        # 构建相机外参矩阵
        camera_extrinsic = self._get_camera_extrinsic(camera_pos, camera_R)
        
        # 将点转换到世界坐标系
        points_world = self._transform_points_to_world(points_cam, camera_extrinsic)
        
        # 创建Open3D点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_world)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        return pcd

    def _get_ee_camera_params(self):
        """
        获取末端执行器相机位置和旋转矩阵
        
        返回:
        camera_pos: 世界坐标系中的相机位置
        camera_R: 相机旋转矩阵（从相机到世界坐标系）
        """
        # 末端执行器姿态
        ee_pos, ee_orn = self.sim.robot.get_ee_pose()
        
        # 末端执行器旋转矩阵
        ee_R = np.array(p.getMatrixFromQuaternion(ee_orn)).reshape(3, 3)
        print("末端执行器方向矩阵:")
        print(ee_R)
        # 相机参数
        cam_cfg = self.config["world_settings"]["camera"]
        ee_offset = np.array(cam_cfg["ee_cam_offset"])
        ee_cam_orn = cam_cfg["ee_cam_orientation"]
        ee_cam_R = np.array(p.getMatrixFromQuaternion(ee_cam_orn)).reshape(3, 3)
        # 计算相机位置
        camera_pos = ee_pos # 为什么ee_pos + ee_R @ ee_offset会错误？
        # 计算相机旋转矩阵
        camera_R = ee_R @ ee_cam_R
        
        return camera_pos, camera_R

    def visualize_point_clouds(self, collected_data, show_frames=True, show_merged=True):
        """
        使用Open3D可视化收集的点云
        
        参数:
        collected_data: 包含点云数据的字典列表
        show_frames: 是否显示坐标系
        show_merged: 是否显示合并点云
        """
        if not collected_data:
            print("没有点云数据可视化")
            return
            
        geometries = []
        
        # 添加世界坐标系
        if show_frames:
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
            geometries.append(coord_frame)
        
        if show_merged:
            # 使用ICP合并点云
            print("使用ICP合并点云...")
            merged_pcd = self.merge_point_clouds(collected_data)
            if merged_pcd is not None:
                # 保留点云的原始颜色
                geometries.append(merged_pcd)
                print(f"添加了合并点云，有 {len(merged_pcd.points)} 个点")
        else:
            # 添加每个点云及其相机坐标系
            for i, data in enumerate(collected_data):
                if 'point_cloud' in data and data['point_cloud'] is not None:
                    # 添加点云
                    geometries.append(data['point_cloud'])
                    
                    # 添加相机坐标系
                    if show_frames:
                        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                        camera_frame.translate(data['camera_position'])
                        camera_frame.rotate(data['camera_rotation'])
                        geometries.append(camera_frame)
                        
                    print(f"添加点云 {i+1}，有 {len(data['point_cloud'].points)} 个点")
        
        print("启动Open3D可视化...")
        o3d.visualization.draw_geometries(geometries)

    def merge_point_clouds(self, collected_data):
        """
        使用ICP配准合并多个点云
        
        参数:
        collected_data: 包含点云数据的字典列表
        
        返回:
        merged_pcd: 合并的点云
        """
        if not collected_data:
            return None
            
        # 使用第一个点云作为参考
        merged_pcd = collected_data[0]['point_cloud']
        
        # ICP参数
        threshold = 0.005  # 距离阈值
        trans_init = np.eye(4)  # 初始变换
        
        # 合并剩余点云
        for i in range(1, len(collected_data)):
            current_pcd = collected_data[i]['point_cloud']
            
            # 执行ICP
            reg_p2p = o3d.pipelines.registration.registration_icp(
                current_pcd, merged_pcd, threshold, trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
            )
            
            # 变换当前点云
            current_pcd.transform(reg_p2p.transformation)
            
            # 合并点云
            merged_pcd += current_pcd
            
            # 可选：使用体素下采样移除重复点
            merged_pcd = merged_pcd.voxel_down_sample(voxel_size=0.005)
            
            print(f"合并点云 {i+1}，适合度: {reg_p2p.fitness}")
        
        return merged_pcd

    def collect_point_clouds(self, target_obj_name=None):
        """
        从多个视点收集点云的主函数
        
        参数:
        target_obj_name: 目标物体名称，如果为None则随机选择
        
        返回:
        collected_data: 包含点云数据的字典列表
        """
        print("开始点云收集...")
        
        # 如果没有指定目标物体名称，随机选择一个
        if target_obj_name is None:
            # 从YCB数据集随机选择一个物体
            object_root_path = self.sim.object_root_path
            files = glob.glob(os.path.join(object_root_path, "Ycb*"))
            obj_names = [os.path.basename(file) for file in files]
            target_obj_name = random.choice(obj_names)
            print(f"使用随机物体重置模拟: {target_obj_name}")
        else:
            print(f"使用指定物体重置模拟: {target_obj_name}")
        
        # 使用目标物体重置模拟
        self.sim.reset(target_obj_name)
        
        # 初始化点云收集列表
        collected_data = []
        
        # 获取并保存仿真环境开始时的初始位置
        initial_joints = self.sim.robot.get_joint_positions()
        print("保存仿真环境初始关节位置")
        
        # 初始化物体高度变量，默认值
        object_height_with_offset = 1.6
        # 初始化物体质心坐标，默认值
        object_centroid_x = -0.02
        object_centroid_y = -0.45

        pause_time = 2.0  # 停顿2秒
        print(f"\n停顿 {pause_time} 秒...")
        for _ in range(int(pause_time * 240)):  # 假设模拟频率为240Hz
            self.sim.step()
            time.sleep(1/240.)
            
        # ===== 移动到指定位置并获取点云 =====
        print("\n移动到高点观察位置...")
        # 定义高点观察位置和方向
        z_observe_pos = np.array([-0.02, -0.45, 1.9])
        z_observe_orn = p.getQuaternionFromEuler([0, np.radians(-180), 0])  # 向下看
        
        # 解算IK
        from src.ik_solver import DifferentialIKSolver
        ik_solver = DifferentialIKSolver(self.sim.robot.id, self.sim.robot.ee_idx, damping=0.05)
        high_point_target_joints = ik_solver.solve(z_observe_pos, z_observe_orn, initial_joints, max_iters=50, tolerance=0.001)
        
        # 生成轨迹
        print("为高点观察位置生成轨迹...")
        high_point_trajectory = TrajectoryPlanner.generate_joint_trajectory(initial_joints, high_point_target_joints, steps=100)
        if not high_point_trajectory:
            print("无法生成到高点观察位置的轨迹，跳过高点点云采集")
        else:
            print(f"生成了包含 {len(high_point_trajectory)} 个点的轨迹")
            
            # 重置到初始位置
            for i, joint_idx in enumerate(self.sim.robot.arm_idx):
                p.resetJointState(self.sim.robot.id, joint_idx, initial_joints[i])
            
            # 沿轨迹移动机器人到高点
            for joint_target in high_point_trajectory:
                self.sim.robot.position_control(joint_target)
                for _ in range(1):
                    self.sim.step()
                    time.sleep(1/240.)
            
            # 在高点观察位置获取点云
            rgb_ee, depth_ee, seg_ee = self.sim.get_ee_renders()
            camera_pos, camera_R = self._get_ee_camera_params()
            print(f"高点观察位置相机位置:", camera_pos)
            print(f"高点观察位置末端执行器位置:", self.sim.robot.get_ee_pose()[0])
            
            # 构建点云
            target_mask_id = self.sim.object.id
            print(f"目标物体ID: {target_mask_id}")
            
            try:
                if target_mask_id not in np.unique(seg_ee):
                    print("警告: 分割掩码中未找到目标物体ID")
                    print("分割掩码中可用的ID:", np.unique(seg_ee))
                    
                    non_zero_ids = np.unique(seg_ee)[1:] if len(np.unique(seg_ee)) > 1 else []
                    if len(non_zero_ids) > 0:
                        target_mask_id = non_zero_ids[0]
                        print(f"使用第一个非零ID代替: {target_mask_id}")
                    else:
                        raise ValueError("分割掩码中没有找到有效物体")
                
                high_point_pcd = self._build_object_point_cloud_ee(rgb_ee, depth_ee, seg_ee, target_mask_id, camera_pos, camera_R)
                
                # 处理点云
                high_point_pcd = high_point_pcd.voxel_down_sample(voxel_size=0.005)
                high_point_pcd, _ = high_point_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
                
                # 存储点云数据
                high_point_cloud_data = {
                    'point_cloud': high_point_pcd,
                    'camera_position': camera_pos,
                    'camera_rotation': camera_R,
                    'ee_position': self.sim.robot.get_ee_pose()[0],
                    'timestamp': time.time(),
                    'target_object': target_obj_name,
                    'viewpoint_idx': 'high_point'
                }
                
                # 获取点云中所有点的坐标
                points_array = np.asarray(high_point_pcd.points)
                if len(points_array) > 0:
                    # 找出z轴最大值点
                    max_z_idx = np.argmax(points_array[:, 2])
                    max_z_point = points_array[max_z_idx]
                    print(f"高点点云中z轴最大值点: {max_z_point}")
                    high_point_cloud_data['max_z_point'] = max_z_point
                    
                    # 提取z轴最大值，加上offset
                    object_max_z = max_z_point[2]
                    object_height_with_offset = max(object_max_z + 0.2, 1.65)
                    print(f"物体高度加偏移量: {object_height_with_offset}")
                    
                    # 计算点云中所有点的x和y坐标质心
                    object_centroid_x = np.mean(points_array[:, 0])
                    object_centroid_y = np.mean(points_array[:, 1])
                    print(f"物体点云质心坐标 (x, y): ({object_centroid_x:.4f}, {object_centroid_y:.4f})")
                    high_point_cloud_data['centroid'] = np.array([object_centroid_x, object_centroid_y, 0])
                else:
                    print("高点点云中没有点")
                
                # 将高点点云添加到收集的数据中
                collected_data.append(high_point_cloud_data)
                print(f"从高点观察位置收集的点云有 {len(high_point_pcd.points)} 个点")
                
            except ValueError as e:
                print(f"为高点观察位置构建点云时出错:", e)

        # 根据物体质心坐标动态生成目标位置和方向
        # 判断物体是否远离机械臂（x<-0.2且y<-0.5视为远离）
        is_object_far = object_centroid_x < -0.2 and object_centroid_y < -0.5
        
        # 基本的采样方向
        target_positions = []
        target_orientations = []
        
        # 东方向
        target_positions.append(np.array([object_centroid_x + 0.15, object_centroid_y, object_height_with_offset]))
        target_orientations.append(p.getQuaternionFromEuler([0, np.radians(-150), 0]))
        
        # 北方向
        target_positions.append(np.array([object_centroid_x, object_centroid_y + 0.15, object_height_with_offset]))
        target_orientations.append(p.getQuaternionFromEuler([np.radians(150), 0, 0]))
        
        # 西方向
        target_positions.append(np.array([object_centroid_x - 0.15, object_centroid_y, object_height_with_offset]))
        target_orientations.append(p.getQuaternionFromEuler([0, np.radians(150), 0]))
        
        # 南方向（如果物体不在远处则添加）
        if not is_object_far:
            target_positions.append(np.array([object_centroid_x, object_centroid_y - 0.15, object_height_with_offset]))
            target_orientations.append(p.getQuaternionFromEuler([np.radians(-150), 0, 0]))
        else:
            print("物体位置较远 (x<-0.2且y<-0.5)，跳过南方向采样点以避免奇异点")
        
        # 顶部视角
        target_positions.append(np.array([-0.02, -0.45, 1.8]))
        target_orientations.append(p.getQuaternionFromEuler([np.radians(180), 0, np.radians(-90)]))
        
        print(f"\n使用基于物体质心的采集位置:")
        print(f"物体质心坐标 (x, y): ({object_centroid_x:.4f}, {object_centroid_y:.4f})")
        print(f"物体高度加偏移量: {object_height_with_offset:.4f}")
        for i, pos in enumerate(target_positions):
            print(f"采集点 {i+1}: ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
        
        # 对每个视点进行采集
        for viewpoint_idx, (target_pos, target_orn) in enumerate(zip(target_positions, target_orientations)):
            print(f"\n移动到视点 {viewpoint_idx + 1}")
            self.sim.get_ee_renders()
            
            # 获取当前关节位置
            current_joints = self.sim.robot.get_joint_positions()
            # 保存当前关节位置
            saved_joints = current_joints.copy()
            
            # 解算目标末端执行器姿态的IK
            target_joints = ik_solver.solve(target_pos, target_orn, current_joints, max_iters=50, tolerance=0.001)
            
            # 重置到保存的起始位置
            for i, joint_idx in enumerate(self.sim.robot.arm_idx):
                p.resetJointState(self.sim.robot.id, joint_idx, saved_joints[i])
            
            # 选择轨迹生成方法
            choice = 2  # 更改此值以测试不同方法
            
            trajectory = []
            if choice == 1:
                print("生成线性笛卡尔轨迹...")
                trajectory = TrajectoryPlanner.generate_cartesian_trajectory(
                    self.sim.robot.id, 
                    self.sim.robot.arm_idx, 
                    self.sim.robot.ee_idx, 
                    saved_joints, 
                    target_pos, 
                    target_orn, 
                    steps=100
                )
            elif choice == 2:
                print("生成线性关节空间轨迹...")
                trajectory = TrajectoryPlanner.generate_joint_trajectory(saved_joints, target_joints, steps=100)
            
            if not trajectory:
                print(f"无法为视点 {viewpoint_idx + 1} 生成轨迹。跳过...")
                continue
            
            print(f"生成了包含 {len(trajectory)} 个点的轨迹")
            
            # 在执行轨迹前再次重置到保存的起始位置
            for i, joint_idx in enumerate(self.sim.robot.arm_idx):
                p.resetJointState(self.sim.robot.id, joint_idx, saved_joints[i])
            
            # 沿轨迹移动机器人到目标位置
            for joint_target in trajectory:
                # 移动机器人
                self.sim.robot.position_control(joint_target)
                for _ in range(1):
                    self.sim.step()
                    time.sleep(1/240.)
            
            # 在此视点捕获点云
            rgb_ee, depth_ee, seg_ee = self.sim.get_ee_renders()
            camera_pos, camera_R = self._get_ee_camera_params()
            print(f"视点 {viewpoint_idx + 1} 相机位置:", camera_pos)
            print(f"视点 {viewpoint_idx + 1} 末端执行器位置:", self.sim.robot.get_ee_pose()[0])
            
            # 构建点云
            target_mask_id = self.sim.object.id
            print(f"目标物体ID: {target_mask_id}")
            
            try:
                if target_mask_id not in np.unique(seg_ee):
                    print("警告: 分割掩码中未找到目标物体ID")
                    print("分割掩码中可用的ID:", np.unique(seg_ee))
                    
                    non_zero_ids = np.unique(seg_ee)[1:] if len(np.unique(seg_ee)) > 1 else []
                    if len(non_zero_ids) > 0:
                        target_mask_id = non_zero_ids[0]
                        print(f"使用第一个非零ID代替: {target_mask_id}")
                    else:
                        raise ValueError("分割掩码中没有找到有效物体")
                
                pcd_ee = self._build_object_point_cloud_ee(rgb_ee, depth_ee, seg_ee, target_mask_id, camera_pos, camera_R)
                
                # 处理点云
                pcd_ee = pcd_ee.voxel_down_sample(voxel_size=0.005)
                pcd_ee, _ = pcd_ee.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
                
                # 存储点云数据
                point_cloud_data = {
                    'point_cloud': pcd_ee,
                    'camera_position': camera_pos,
                    'camera_rotation': camera_R,
                    'ee_position': self.sim.robot.get_ee_pose()[0],
                    'timestamp': time.time(),
                    'target_object': target_obj_name,
                    'viewpoint_idx': viewpoint_idx
                }
                collected_data.append(point_cloud_data)
                print(f"从视点 {viewpoint_idx + 1} 收集的点云有 {len(pcd_ee.points)} 个点。")
                
            except ValueError as e:
                print(f"为视点 {viewpoint_idx + 1} 构建点云时出错:", e)

        return collected_data
