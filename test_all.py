import os
import glob
import yaml
import time
import random
import numpy as np
import pybullet as p
import open3d as o3d
from pybullet_object_models import ycb_objects  # type:ignore
from src.simulation import Simulation
from src.ik_solver import DifferentialIKSolver
from src.obstacle_tracker import ObstacleTracker
from src.path_planning.rrt_star import RRTStarPlanner
# from src.potential_field import PotentialFieldPlanner
from src.grasping.grasping import GraspGeneration
# from src.grasping.grasp_generation import GraspGeneration
from src.grasping import grasping_mesh
from src.point_cloud import object_mesh
from scipy.spatial.transform import Rotation

# Check if PyBullet has NumPy support enabled
numpy_support = p.isNumpyEnabled()
print(f"PyBullet NumPy support enabled: {numpy_support}")

def convert_depth_to_meters(depth_buffer, near, far):
    """
    convert depth buffer values to actual distance (meters)
    
    Parameters:
    depth_buffer: depth buffer values obtained from PyBullet
    near, far: near/far plane distances
    
    Returns:
    actual depth values in meters
    """
    return far * near / (far - (far - near) * depth_buffer)

def get_camera_intrinsic(width, height, fov):
    """
    calculate intrinsic matrix from camera parameters
    
    Parameters:
    width: image width (pixels)
    height: image height (pixels)
    fov: vertical field of view (degrees)

    Returns:
    camera intrinsic matrix
    """    
    # calculate focal length
    f = height / (2 * np.tan(np.radians(fov / 2)))
    
    # calculate principal point
    cx = width / 2
    cy = height / 2
    
    intrinsic_matrix = np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0, 1]
    ])
    
    return intrinsic_matrix

def depth_image_to_point_cloud(depth_image, mask, rgb_image, intrinsic_matrix):
    """
    depth image to camera coordinate point cloud
    
    Parameters:
    depth_image: depth image (meters)
    mask: target object mask (boolean array)
    rgb_image: RGB image
    intrinsic_matrix: camera intrinsic matrix
    
    Returns:
    camera coordinate point cloud (N,3) and corresponding colors (N,3)
    """
    # extract pixel coordinates of target mask
    rows, cols = np.where(mask)
    
    if len(rows) == 0:
        raise ValueError("No valid pixels found in target mask")
    
    # extract depth values of these pixels
    depths = depth_image[rows, cols]
    
    # image coordinates to camera coordinates
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    
    # calculate camera coordinates
    x = -(cols - cx) * depths / fx # negative sign due to PyBullet camera orientation???
    y = -(rows - cy) * depths / fy
    z = depths
    
    # stack points
    points = np.vstack((x, y, z)).T
    
    # extract RGB colors
    colors = rgb_image[rows, cols, :3].astype(np.float64) / 255.0
    
    return points, colors

def transform_points_to_world(points, camera_extrinsic):
    """
    transform points from camera coordinates to world coordinates
    
    Parameters:
    points: point cloud in camera coordinates (N,3)
    camera_extrinsic: camera extrinsic matrix (4x4)
    
    Returns:
    point cloud in world coordinates (N,3)
    """
    # convert point cloud to homogeneous coordinates
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    
    # transform point cloud using extrinsic matrix
    world_points_homogeneous = (camera_extrinsic @ points_homogeneous.T).T # points in rows
    
    # convert back to non-homogeneous coordinates
    world_points = world_points_homogeneous[:, :3]
    
    return world_points

def get_camera_extrinsic(camera_pos, camera_R):
    """
    build camera extrinsic matrix (transform from camera to world coordinates)
    
    Parameters:
    camera_pos: camera position in world coordinates
    camera_R: camera rotation matrix (3x3)
    
    Returns:
    camera extrinsic matrix (4x4)
    """
    # build 4x4 extrinsic matrix
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = camera_R
    extrinsic[:3, 3] = camera_pos
    
    return extrinsic

def build_object_point_cloud_ee(rgb, depth, seg, target_mask_id, config, camera_pos, camera_R):
    """
    build object point cloud using end-effector camera RGB, depth, segmentation data
    
    Parameters:
    rgb: RGB image
    depth: depth buffer values
    seg: segmentation mask
    target_mask_id: target object ID
    config: configuration dictionary
    camera_pos: camera position in world coordinates
    camera_R: camera rotation matrix (from camera to world coordinates)
    
    Returns:
    Open3D point cloud object
    """
    # read camera parameters
    cam_cfg = config["world_settings"]["camera"]
    width = cam_cfg["width"]
    height = cam_cfg["height"]
    fov = cam_cfg["fov"]  # vertical FOV
    near = cam_cfg["near"]
    far = cam_cfg["far"]
    
    # create target object mask
    object_mask = (seg == target_mask_id)
    if np.count_nonzero(object_mask) == 0:
        raise ValueError(f"Target mask ID {target_mask_id} not found in segmentation.")
    
    # extract depth buffer values for target object
    metric_depth = convert_depth_to_meters(depth, near, far)
    
    # get intrinsic matrix
    intrinsic_matrix = get_camera_intrinsic(width, height, fov)
    
    # convert depth image to point cloud
    points_cam, colors = depth_image_to_point_cloud(metric_depth, object_mask, rgb, intrinsic_matrix)
    
    # build camera extrinsic matrix
    camera_extrinsic = get_camera_extrinsic(camera_pos, camera_R)
    
    # transform points to world coordinates
    points_world = transform_points_to_world(points_cam, camera_extrinsic)
    
    # create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_world)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd

def get_ee_camera_params(robot, config):
    """
    get end-effector camera position and rotation matrix
    
    Parameters:
    robot: robot object
    config: configuration dictionary
    
    Returns:
    camera_pos: camera position in world coordinates
    camera_R: camera rotation matrix (from camera to world coordinates)
    """
    # end-effector pose
    ee_pos, ee_orn = robot.get_ee_pose()
    
    # end-effector rotation matrix
    ee_R = np.array(p.getMatrixFromQuaternion(ee_orn)).reshape(3, 3)
    print("End effector orientation matrix:")
    print(ee_R)
    # camera parameters
    cam_cfg = config["world_settings"]["camera"]
    ee_offset = np.array(cam_cfg["ee_cam_offset"])
    ee_cam_orn = cam_cfg["ee_cam_orientation"]
    ee_cam_R = np.array(p.getMatrixFromQuaternion(ee_cam_orn)).reshape(3, 3)
    # calculate camera position
    camera_pos = ee_pos # why ee_pos + ee_R @ ee_offset will be wrong?
    # calculate camera rotation matrix
    camera_R = ee_R @ ee_cam_R
    
    return camera_pos, camera_R

# for linear trajectory in Cartesian space
def generate_cartesian_trajectory(sim, ik_solver, start_joints, target_pos, target_orn, steps=100):
    """
    generate linear Cartesian trajectory in Cartesian space
    """
    # set start position
    for i, joint_idx in enumerate(sim.robot.arm_idx):
        p.resetJointState(sim.robot.id, joint_idx, start_joints[i])
    
    # get current end-effector pose
    ee_state = p.getLinkState(sim.robot.id, sim.robot.ee_idx)
    print(f"ee_state_0={np.array(ee_state[0])}, ee_state_1={np.array(ee_state[1])}")
    start_pos = np.array(ee_state[0])
    
    # generate linear trajectory
    trajectory = []
    for step in range(steps + 1):
        t = step / steps  # normalize step
        
        # linear interpolation
        pos = start_pos + t * (target_pos - start_pos)
        
        # solve IK for current Cartesian position
        current_joints = ik_solver.solve(pos, target_orn, start_joints, max_iters=50, tolerance=0.001)
        
        # add solution to trajectory
        trajectory.append(current_joints)
        
        # reset to start position
        for i, joint_idx in enumerate(sim.robot.arm_idx):
            p.resetJointState(sim.robot.id, joint_idx, start_joints[i])
    
    return trajectory

# for trajectory in joint space
def generate_trajectory(start_joints, end_joints, steps=100):
    """
    generate smooth trajectory from start to end joint positions
    
    Parameters:
    start_joints: start joint positions
    end_joints: end joint positions
    steps: number of steps for interpolation
    
    Returns:
    trajectory: list of joint positions
    """
    trajectory = []
    for step in range(steps + 1):
        t = step / steps  # normalize step
        # linear interpolation
        point = [start + t * (end - start) for start, end in zip(start_joints, end_joints)]
        trajectory.append(point)
    return trajectory

def generate_rrt_star_trajectory(sim, rrt_planner, start_joints, target_joints, visualize=True):
    """
    Generate a collision-free trajectory using RRT* planning.
    
    Args:
        sim: Simulation instance
        rrt_planner: RRTStarPlanner instance
        start_joints: Start joint configuration
        target_joints: Target joint configuration
        visualize: Whether to visualize the planning process
        
    Returns:
        Smooth trajectory as list of joint configurations
    """
    print("Planning path with RRT*...")
    
    # Plan path using RRT*
    path, path_cost = rrt_planner.plan(start_joints, target_joints)
    
    if not path:
        print("Failed to find a valid path!")
        return []
    
    print(f"Path found with {len(path)} waypoints and cost {path_cost:.4f}")
    
    # Generate smooth trajectory
    trajectory = rrt_planner.generate_smooth_trajectory(path, smoothing_steps=20)
    
    print(f"Generated smooth trajectory with {len(trajectory)} points")
    
    # Visualize the path if requested
    if visualize:
        # Clear previous visualization
        rrt_planner.clear_visualization()
        
        # Visualize the path
        for i in range(len(path) - 1):
            start_ee, _ = rrt_planner._get_current_ee_pose(path[i])
            end_ee, _ = rrt_planner._get_current_ee_pose(path[i+1])
            
            p.addUserDebugLine(
                start_ee, end_ee, [0, 0, 1], 3, 0)
            
    return trajectory

def visualize_point_clouds(collected_data, show_frames=True, show_merged=True):
    """
    Visualize collected point clouds using Open3D
    
    Parameters:
    collected_data: list of dictionaries containing point cloud data
    show_frames: whether to show coordinate frames
    show_merged: whether to show merged point cloud
    """
    if not collected_data:
        print("No point cloud data to visualize")
        return
        
    geometries = []
    
    # Add world coordinate frame
    if show_frames:
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
        geometries.append(coord_frame)
    
    if show_merged:
        # Merge point clouds using ICP
        print("Merging point clouds using ICP...")
        merged_pcd = iterative_closest_point(collected_data)
        if merged_pcd is not None:
            # Keep original colors from point clouds
            geometries.append(merged_pcd)
            print(f"Added merged point cloud with {len(merged_pcd.points)} points")
    else:
        # Add each point cloud and its camera frame
        for i, data in enumerate(collected_data):
            if 'point_cloud' in data and data['point_cloud'] is not None:
                # Add point cloud
                geometries.append(data['point_cloud'])
                
                # Add camera frame
                if show_frames:
                    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                    camera_frame.translate(data['camera_position'])
                    camera_frame.rotate(data['camera_rotation'])
                    geometries.append(camera_frame)
                    
                print(f"Added point cloud {i+1} with {len(data['point_cloud'].points)} points")
    
    print("Launching Open3D visualization...")
    o3d.visualization.draw_geometries(geometries)

def iterative_closest_point(collected_data):
    """
    Merge multiple point clouds using ICP registration
    
    Parameters:
    collected_data: list of dictionaries containing point cloud data
    
    Returns:
    merged_pcd: merged point cloud
    """
    if not collected_data:
        return None
        
    # Use the first point cloud as reference
    merged_pcd = collected_data[0]['point_cloud']
    
    # ICP parameters
    threshold = 0.005  # distance threshold
    trans_init = np.eye(4)  # initial transformation
    
    # Merge remaining point clouds
    for i in range(1, len(collected_data)):
        current_pcd = collected_data[i]['point_cloud']
        
        # Perform ICP
        reg_p2p = o3d.pipelines.registration.registration_icp(
            current_pcd, merged_pcd, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
        )
        
        # Transform current point cloud
        current_pcd.transform(reg_p2p.transformation)
        
        # Merge point clouds
        merged_pcd += current_pcd
        
        # Optional: Remove duplicates using voxel downsampling
        merged_pcd = merged_pcd.voxel_down_sample(voxel_size=0.005)
        
        print(f"Merged point cloud {i+1}, fitness: {reg_p2p.fitness}")
    
    return merged_pcd

def run_grasping(config, sim, collected_point_clouds):
    """
    执行抓取生成和可视化
    
    参数:
    config: 配置字典
    sim: 模拟对象
    collected_point_clouds: 收集的点云数据列表
    """
    # 打开文件用于保存grasp_center
    grasp_center_file = open("paste.txt", "w")
    
    print("合并点云并计算质心...")
    merged_pcd = iterative_closest_point(collected_point_clouds)
    centre_point = np.asarray(merged_pcd.points)
    centre_point = centre_point.mean(axis=0)
    print(f"点云质心坐标: {centre_point}")
    
    # 在PyBullet中可视化点云质心
    print("在PyBullet中可视化点云质心...")
    # 创建一个红色球体表示质心
    visual_id = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=0.01,  # 1cm半径的球体
        rgbaColor=[1, 0, 0, 1]  # 红色
    )
    centroid_id = p.createMultiBody(
        baseMass=0,  # 质量为0表示静态物体
        baseVisualShapeIndex=visual_id,
        basePosition=centre_point.tolist()
    )
    
    # 添加文本标签
    p.addUserDebugText(
        f"Centroid ({centre_point[0]:.3f}, {centre_point[1]:.3f}, {centre_point[2]:.3f})",
        centre_point + np.array([0, 0, 0.05]),  # 在质心上方5cm处显示文本
        [1, 1, 1],  # 白色文本
        1.0  # 文本大小
    )
    
    # 计算点云的边界框和高度
    points = np.asarray(merged_pcd.points)
    
    # 实现XY平面旋转的最小体积边界框(OBB)
    # 1. 将点云投影到XY平面
    points_xy = points.copy()
    points_xy[:, 2] = 0  # 将Z坐标设为0，投影到XY平面
    
    # 2. 对XY平面上的点云进行PCA，找到主轴方向
    xy_mean = np.mean(points_xy, axis=0)
    xy_centered = points_xy - xy_mean
    cov_xy = np.cov(xy_centered.T)[:2, :2]  # 只取XY平面的协方差
    eigenvalues, eigenvectors = np.linalg.eigh(cov_xy)
    
    # 排序特征值和特征向量（降序）
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # 3. 获取主轴方向，这些是XY平面内的旋转方向
    main_axis_x = np.array([eigenvectors[0, 0], eigenvectors[1, 0], 0])
    main_axis_y = np.array([eigenvectors[0, 1], eigenvectors[1, 1], 0])
    main_axis_z = np.array([0, 0, 1])  # Z轴保持垂直
    
    # 归一化主轴
    main_axis_x = main_axis_x / np.linalg.norm(main_axis_x)
    main_axis_y = main_axis_y / np.linalg.norm(main_axis_y)
    
    # 4. 构建旋转矩阵
    rotation_matrix = np.column_stack((main_axis_x, main_axis_y, main_axis_z))
    
    # 5. 将点云旋转到新坐标系
    points_rotated = np.dot(points - xy_mean, rotation_matrix)
    
    # 6. 在新坐标系中计算边界框
    min_point_rotated = np.min(points_rotated, axis=0)
    max_point_rotated = np.max(points_rotated, axis=0)
    
    # 计算旋转后的边界框尺寸
    obb_dims = max_point_rotated - min_point_rotated
    object_height = obb_dims[2]
    
    # 计算边界框的8个顶点（在旋转坐标系中）
    bbox_corners_rotated = np.array([
        [min_point_rotated[0], min_point_rotated[1], min_point_rotated[2]],
        [max_point_rotated[0], min_point_rotated[1], min_point_rotated[2]],
        [max_point_rotated[0], max_point_rotated[1], min_point_rotated[2]],
        [min_point_rotated[0], max_point_rotated[1], min_point_rotated[2]],
        [min_point_rotated[0], min_point_rotated[1], max_point_rotated[2]],
        [max_point_rotated[0], min_point_rotated[1], max_point_rotated[2]],
        [max_point_rotated[0], max_point_rotated[1], max_point_rotated[2]],
        [min_point_rotated[0], max_point_rotated[1], max_point_rotated[2]],
    ])
    
    # 将顶点变换回原始坐标系
    bbox_corners = np.dot(bbox_corners_rotated, rotation_matrix.T) + xy_mean
    
    # 计算轴对齐边界框(AABB)用于抓取采样（基于OBB的顶点）
    aabb_min_point = np.min(bbox_corners, axis=0)
    aabb_max_point = np.max(bbox_corners, axis=0)
    
    print(f"物体高度: {object_height:.4f}m")
    print(f"物体边界框尺寸(旋转坐标系): {obb_dims}")
    print(f"主轴方向X: {main_axis_x}")
    print(f"主轴方向Y: {main_axis_y}")
    print(f"轴对齐边界框最小点: {aabb_min_point}")
    print(f"轴对齐边界框最大点: {aabb_max_point}")
    
    # 可视化物体边界框
    bbox_lines = [
        # 底部矩形
        [bbox_corners[0], bbox_corners[1]],
        [bbox_corners[1], bbox_corners[2]],
        [bbox_corners[2], bbox_corners[3]],
        [bbox_corners[3], bbox_corners[0]],
        # 顶部矩形
        [bbox_corners[4], bbox_corners[5]],
        [bbox_corners[5], bbox_corners[6]],
        [bbox_corners[6], bbox_corners[7]],
        [bbox_corners[7], bbox_corners[4]],
        # 连接线
        [bbox_corners[0], bbox_corners[4]],
        [bbox_corners[1], bbox_corners[5]],
        [bbox_corners[2], bbox_corners[6]],
        [bbox_corners[3], bbox_corners[7]]
    ]
    
    for line in bbox_lines:
        p.addUserDebugLine(
            line[0], 
            line[1], 
            [0, 1, 1],  # 青色
            1, 
            0
        )
    
    # 初始化IK求解器
    ik_solver = DifferentialIKSolver(sim.robot.id, sim.robot.ee_idx, damping=0.05)
    
    # 获取当前关节位置
    current_joints = sim.robot.get_joint_positions()
    
    # 初始化抓取生成器
    print("生成抓取候选...")
    grasp_generator = GraspGeneration()
    
    sampled_grasps = grasp_generator.sample_grasps(
        centre_point, 
        num_grasps=500, 
        radius=0.1, 
        sim=sim,
        rotation_matrix=rotation_matrix,
        min_point_rotated=min_point_rotated,
        max_point_rotated=max_point_rotated,
        center_rotated=xy_mean
    )
    
    # 为每个抓取创建网格
    all_grasp_meshes = []
    for grasp in sampled_grasps:
        R, grasp_center = grasp
        all_grasp_meshes.append(grasping_mesh.create_grasp_mesh(center_point=grasp_center, rotation_matrix=R))

    # 从点云创建三角网格用于可视化
    print("从点云创建三角网格...")
    obj_triangle_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd=merged_pcd, 
                                                                                      alpha=0.08)
 
    # 评估抓取质量
    print("评估抓取质量...")
    vis_meshes = [obj_triangle_mesh]
    highest_quality = 0
    highest_containment_grasp = None
    best_grasp = None
    
    for (pose, grasp_mesh) in zip(sampled_grasps, all_grasp_meshes):
        print(f"grasp mesh:{grasp_mesh}")
        if not grasp_generator.check_grasp_collision(grasp_mesh, object_pcd=merged_pcd, num_colisions=1):
            # 输出每个有效grasp_center并保存到文件
            R, grasp_center = pose
            print(f"grasp_center: {grasp_center}")
            grasp_center_file.write(f"grasp_center: {grasp_center}\n")
            
            valid_grasp, grasp_quality, max_interception_depth = grasp_generator.check_grasp_containment(
                grasp_mesh[0].get_center(), 
                grasp_mesh[1].get_center(),
                finger_length=0.05,
                object_pcd=merged_pcd,
                num_rays=50,
                rotation_matrix=pose[0],
                visualize_rays=False # toggle visualization of ray casting
            )
            
            # 使用新的质量指标选择抓取
            if valid_grasp and grasp_quality > highest_quality:
                highest_quality = grasp_quality
                highest_containment_grasp = grasp_mesh
                best_grasp = pose
                print(f"找到更好的抓取，质量: {grasp_quality:.3f}")

    # 可视化最佳抓取
    if highest_containment_grasp is not None:
        print(f"找到有效抓取，最高质量: {highest_quality:.3f}")
        vis_meshes.extend(highest_containment_grasp)
        
        # 打印最佳抓取姿态信息
        print("\n========== 最佳抓取姿态信息 ==========")
        R, grasp_center = best_grasp
        print(f"抓取中心位置: {grasp_center}")
        
        # 构建爪子自身坐标系中的偏移向量
        # 根据create_grasp_mesh中的坐标系定义，y轴是手指延伸方向
        local_offset = np.array([0, 0.06, 0])

        # 使用旋转矩阵将偏移向量从爪子坐标系转换到世界坐标系
        world_offset = R @ local_offset

        # 计算补偿后的末端执行器目标位置
        ee_target_pos = grasp_center + world_offset
        
        print(f"原始grasp_center: {grasp_center}")
        print(f"补偿向量: {world_offset}")
        print(f"补偿后ee_target_pos: {ee_target_pos}")
        
        # 将旋转矩阵转换为四元数
        rot = Rotation.from_matrix(R)
        quat = rot.as_quat()  # [x, y, z, w]格式
        euler = rot.as_euler('xyz', degrees=True)  # 欧拉角（度）
        
        print(f"抓取旋转矩阵:\n{R}")
        print(f"抓取四元数 [x, y, z, w]: {quat}")
        print(f"抓取欧拉角 [x, y, z] (度): {euler}")
        
        # 计算末端执行器的目标位姿
        # 注意：这里假设抓取中心就是末端执行器的目标位置，旋转矩阵就是末端执行器的目标方向
        # 实际应用中可能需要根据机器人的具体配置进行调整
        ee_target_orn = p.getQuaternionFromEuler([euler[0]/180*np.pi, euler[1]/180*np.pi, euler[2]/180*np.pi])
        
        print("\n========== 末端执行器目标位姿 ==========")
        print(f"位置: {ee_target_pos}")
        print(f"四元数 [x, y, z, w]: {ee_target_orn}")
        
        # 添加坐标系转换，使打印的姿态与Open3D可视化一致
        combined_transform = np.array([
            [0, -1, 0],
            [0, 0, -1],
            [1, 0, 0]
        ])
        # combined_transform = np.array([
        #     [0, 1, 0],
        #     [0, 0, -1],
        #     [-1, 0, 0]
        # ])
        
        # 直接应用合并后的转换
        R_world = R @ combined_transform
        
        rot_world = Rotation.from_matrix(R_world)
        quat_world = rot_world.as_quat()
        euler_world = rot_world.as_euler('xyz', degrees=True)
        
        print("\n========== 转换后的末端执行器位姿 ==========")
        print(f"位置: {ee_target_pos}")
        print(f"旋转矩阵:\n{R_world}")
        print(f"四元数 [x, y, z, w]: {quat_world}")
        print(f"欧拉角 [x, y, z] (度): {euler_world}")
        
        # ===== 添加机械臂运动规划代码 =====
        print("\n========== 规划机械臂运动轨迹 ==========")
        
        # 定义pose 2（最终抓取位姿）
        pose2_pos = ee_target_pos
        pose2_orn = p.getQuaternionFromEuler([euler_world[0]/180*np.pi, euler_world[1]/180*np.pi, euler_world[2]/180*np.pi])
        
        # 计算pose 1（抓取前位置）- 沿着pose 2自身的z轴往后退0.05米
        # 获取pose 2的旋转矩阵，用于计算z轴方向
        pose2_rot_matrix = R_world
        
        # 提取旋转矩阵的第三列，即z轴方向
        z_axis = pose2_rot_matrix[:, 2]
        
        # 沿z轴方向后退0.15米
        pose1_pos = pose2_pos - 0.15 * z_axis
        pose1_orn = pose2_orn  # 保持相同的方向
        
        print(f"Pose 2（最终抓取位姿）- 位置: {pose2_pos}, 方向: {pose2_orn}")
        print(f"Pose 1（抓取前位置）- 位置: {pose1_pos}, 方向: {pose1_orn}")
        
        # ===== 在PyBullet中可视化pose 1和pose 2的坐标轴 =====
        print("\n========== 可视化抓取位姿坐标轴 ==========")
        
        # 坐标轴长度
        axis_length = 0.1
        
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
        
        # 获取当前机械臂关节角度（点云采集的最后一点）
        start_joints = sim.robot.get_joint_positions()
        print(f"起始关节角度: {start_joints}")
        
        # 解算Pose 1的IK
        target_joints = ik_solver.solve(pose1_pos, pose1_orn, start_joints, max_iters=50, tolerance=0.001)
        
        if target_joints is not None:
            print(f"目标关节角度: {target_joints}")
            
            # 生成从当前位置到Pose 1的轨迹
            print("生成轨迹...")
            trajectory = generate_trajectory(start_joints, target_joints, steps=100)
            
            if trajectory:
                print(f"生成了包含 {len(trajectory)} 个点的轨迹")
                
                # 执行轨迹
                print("执行轨迹...")
                for joint_target in trajectory:
                    sim.robot.position_control(joint_target)
                    for _ in range(1):
                        sim.step()
                        time.sleep(1/240.)
                
                print("机械臂已移动到抓取前位置")
                
                # ===== 打开爪子 =====
                print("\n========== 打开爪子 ==========")
                # 根据配置文件，gripper_idx是[9, 10]，默认值是[0.02, 0.02]
                # 设置较大的值打开爪子
                open_gripper_width = 0.04  # 打开爪子的宽度
                p.setJointMotorControlArray(
                    sim.robot.id,
                    jointIndices=sim.robot.gripper_idx,
                    controlMode=p.POSITION_CONTROL,
                    targetPositions=[open_gripper_width, open_gripper_width]
                )
                
                # 等待爪子打开
                for _ in range(int(0.5 * 240)):  # 等待0.5秒
                    sim.step()
                    time.sleep(1/240.)
                
                print("爪子已打开")
                
                # ===== 移动到Pose 2（最终抓取位置） =====
                print("\n========== 移动到最终抓取位置 ==========")
                # 获取当前关节角度
                current_joints = sim.robot.get_joint_positions()
                
                # 解算Pose 2的IK
                pose2_target_joints = ik_solver.solve(pose2_pos, pose2_orn, current_joints, max_iters=50, tolerance=0.001)
                
                if pose2_target_joints is not None:
                    print(f"Pose 2目标关节角度: {pose2_target_joints}")
                    
                    # 生成从Pose 1到Pose 2的轨迹
                    print("生成到Pose 2的轨迹...")
                    pose2_trajectory = generate_cartesian_trajectory(sim, ik_solver, current_joints, pose2_pos, pose2_orn, steps=50) # has to be straight line
                    
                    if pose2_trajectory:
                        print(f"生成了包含 {len(pose2_trajectory)} 个点的轨迹")
                        
                        # 执行轨迹
                        print("执行轨迹...")
                        for joint_target in pose2_trajectory:
                            sim.robot.position_control(joint_target)
                            for _ in range(1):
                                sim.step()
                                time.sleep(1/240.)
                        
                        print("机械臂已移动到最终抓取位置")
                        
                        for _ in range(int(0.5 * 240)):  # 等待1秒
                            sim.step()
                            time.sleep(1/240.)
                            
                        # ===== 关闭爪子进行抓取 =====
                        print("\n========== 关闭爪子抓取物体 ==========")
                        # 设置较小的值关闭爪子
                        close_gripper_width = 0.005  # 关闭爪子的宽度
                        p.setJointMotorControlArray(
                            sim.robot.id,
                            jointIndices=sim.robot.gripper_idx,
                            controlMode=p.POSITION_CONTROL,
                            targetPositions=[close_gripper_width, close_gripper_width]
                        )
                        
                        # 等待爪子关闭
                        for _ in range(int(1.0 * 240)):  # 等待1秒
                            sim.step()
                            time.sleep(1/240.)
                        
                        print("爪子已关闭，物体已抓取")
                        
                        # ===== 抓取后向上提升物体 =====
                        print("\n========== 向上提升物体 ==========")
                        # 获取当前末端执行器位置和方向
                        current_ee_pos, current_ee_orn = sim.robot.get_ee_pose()
                        print(f"当前末端执行器位置: {current_ee_pos}")
                        
                        # 计算向上提升后的位置（世界坐标系z轴方向）
                        lift_height = 0.5  # 提升高度（米）
                        lift_pos = current_ee_pos.copy()
                        lift_pos[2] += lift_height  # 在z轴方向上增加高度
                        
                        print(f"提升后的目标位置: {lift_pos}")
                        
                        # 获取当前关节角度
                        current_joints = sim.robot.get_joint_positions()
                        
                        # 解算提升位置的IK
                        lift_target_joints = ik_solver.solve(lift_pos, current_ee_orn, current_joints, max_iters=50, tolerance=0.001)
                        
                        if lift_target_joints is not None:
                            print(f"提升位置的目标关节角度: {lift_target_joints}")
                            
                            # 生成从当前位置到提升位置的轨迹
                            print("生成提升轨迹...")
                            lift_trajectory = generate_trajectory(current_joints, lift_target_joints, steps=100)
                            
                            if lift_trajectory:
                                print(f"生成了包含 {len(lift_trajectory)} 个点的提升轨迹")
                                
                                # 执行轨迹
                                print("执行提升轨迹...")
                                for joint_target in lift_trajectory:
                                    sim.robot.position_control(joint_target)
                                    for _ in range(5):
                                        sim.step()
                                        time.sleep(1/240.)
                                
                                print("物体已成功提升")
                            else:
                                print("无法生成提升轨迹")
                        else:
                            print("无法解算提升位置的IK，无法提升物体")
                    else:
                        print("无法生成到Pose 2的轨迹")
                else:
                    print("无法解算Pose 2的IK，无法移动到最终抓取位置")
            else:
                print("无法生成轨迹")
        else:
            print("无法解算IK，无法移动到抓取前位置")
    else:
        print("未找到有效抓取!")

    # 显示可视化结果
    print("显示抓取可视化结果...")
    object_mesh.visualize_3d_objs(vis_meshes)
    
    # 关闭文件
    grasp_center_file.close()

def run_planning(config, sim):
    """
    使用RRT*规划从抓取后上举位置到托盘位置的路径
    
    参数:
    config: 配置字典
    sim: 模拟对象
    """
    print("\n========== 开始路径规划 ==========")
    
    # 初始化障碍物跟踪器
    obstacle_tracker = ObstacleTracker(n_obstacles=2, exp_settings=config)
    
    # 使用静态相机获取障碍物位置
    print("获取障碍物信息...")
    rgb_static, depth_static, seg_static = sim.get_static_renders()
    detections = obstacle_tracker.detect_obstacles(rgb_static, depth_static, seg_static)
    tracked_positions = obstacle_tracker.update(detections)
    
    # 可视化障碍物边界框
    bounding_box_ids = obstacle_tracker.visualize_tracking_3d(tracked_positions)
    print(f"检测到 {len(tracked_positions)} 个障碍物")
    
    # 获取当前机械臂位置（抓取后上举位置）
    current_joints = sim.robot.get_joint_positions()
    current_ee_pos, current_ee_orn = sim.robot.get_ee_pose()
    print(f"当前末端执行器位置: {current_ee_pos}")
    
    # 计算托盘位置（目标位置）
    min_lim, max_lim = sim.goal._get_goal_lims()
    goal_pos = np.array([
        (min_lim[0] + max_lim[0])/2,
        (min_lim[1] + max_lim[1])/2,
        max_lim[2] + 0.2
    ])
    # goal_pos[0] -= 0.2
    # goal_pos[1] -= 0.2
    print(f"托盘目标位置: {goal_pos}")
    
    # 在PyBullet中可视化托盘目标位置
    visual_id = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=0.03,  # 3cm半径的球体
        rgbaColor=[0, 0, 1, 0.7]  # 蓝色半透明
    )
    goal_marker_id = p.createMultiBody(
        baseMass=0,  # 质量为0表示静态物体
        baseVisualShapeIndex=visual_id,
        basePosition=goal_pos.tolist()
    )
    
    # 添加目标位置坐标轴
    axis_length = 0.1  # 10cm长的坐标轴
    p.addUserDebugLine(
        goal_pos, 
        goal_pos + np.array([axis_length, 0, 0]), 
        [1, 0, 0], 3, 0  # X轴 - 红色
    )
    p.addUserDebugLine(
        goal_pos, 
        goal_pos + np.array([0, axis_length, 0]), 
        [0, 1, 0], 3, 0  # Y轴 - 绿色
    )
    p.addUserDebugLine(
        goal_pos, 
        goal_pos + np.array([0, 0, axis_length]), 
        [0, 0, 1], 3, 0  # Z轴 - 蓝色
    )
    
    # 添加目标位置文本标签
    p.addUserDebugText(
        f"Goal Position ({goal_pos[0]:.3f}, {goal_pos[1]:.3f}, {goal_pos[2]:.3f})",
        goal_pos + np.array([0, 0, 0.05]),  # 在目标位置上方5cm处显示文本
        [1, 1, 1],  # 白色文本
        1.0  # 文本大小
    )
    
    # 初始化IK求解器
    ik_solver = DifferentialIKSolver(sim.robot.id, sim.robot.ee_idx, damping=0.05)
    
    # 解算目标位置的IK
    goal_orn = current_ee_orn  # 保持当前方向
    # goal_orn = p.getQuaternionFromEuler([np.radians(-90), np.radians(-90), np.radians(-45)])
    goal_joints = ik_solver.solve(goal_pos, goal_orn, current_joints, max_iters=50, tolerance=0.001)
    
    if goal_joints is None:
        print("无法解算目标位置的IK，无法进行路径规划")
        return
    
    print("初始化RRT*规划器...")
    # 初始化RRT*规划器
    rrt_planner = RRTStarPlanner(
        robot_id=sim.robot.id,
        joint_indices=sim.robot.arm_idx,
        lower_limits=sim.robot.lower_limits,
        upper_limits=sim.robot.upper_limits,
        ee_link_index=sim.robot.ee_idx,
        obstacle_tracker=obstacle_tracker,
        max_iterations=3000,  # 增加迭代次数以获得更好的路径
        step_size=0.2,
        goal_sample_rate=0.1,  # 增加目标采样率
        search_radius=0.6,
        goal_threshold=0.01,
        collision_check_step=0.05
    )
    
    print("开始RRT*路径规划...")
    # 规划路径
    trajectory = generate_rrt_star_trajectory(sim, rrt_planner, current_joints, goal_joints, visualize=True)
    
    if not trajectory:
        print("无法生成到目标位置的路径，规划失败")
        return
    
    print(f"生成了包含 {len(trajectory)} 个点的轨迹")
    
    # 直接执行RRT*生成的轨迹
    print("执行轨迹...")
    for joint_target in trajectory:
        sim.robot.position_control(joint_target)
        for _ in range(5):
            sim.step()
            time.sleep(1/240.)
    
    print("机械臂已成功移动到托盘位置")
    
    # 放下物体
    print("\n========== 放下物体 ==========")
    # 打开爪子
    open_gripper_width = 0.04  # 打开爪子的宽度
    p.setJointMotorControlArray(
        sim.robot.id,
        jointIndices=sim.robot.gripper_idx,
        controlMode=p.POSITION_CONTROL,
        targetPositions=[open_gripper_width, open_gripper_width]
    )
    
    # 等待爪子打开
    for _ in range(int(1.0 * 240)):  # 等待1秒
        sim.step()
        time.sleep(1/240.)
    
    print("爪子已打开，物体已放置到托盘位置")

def run_pcd(config):
    """
    main function to run point cloud collection from multiple viewpoints
    """
    print("Starting point cloud collection ...")
    
    # initialize PyBullet simulation
    
    # randomly select an object from YCB dataset
    object_root_path = ycb_objects.getDataPath()
    files = glob.glob(os.path.join(object_root_path, "Ycb*"))
    obj_names = [os.path.basename(file) for file in files]
    # target_obj_name = random.choice(obj_names)
    # print(f"Resetting simulation with random object: {target_obj_name}")
    # All objects: 
    # Low objects: YcbBanana, YcbFoamBrick, YcbHammer, YcbMediumClamp, YcbPear, YcbScissors, YcbStrawberry, YcbTennisBall, 
    # Medium objects: YcbGelatinBox, YcbMasterChefCan, YcbPottedMeatCan, YcbTomatoSoupCan
    # High objects: YcbCrackerBox, YcbMustardBottle, 
    # Unstable objects: YcbChipsCan, YcbPowerDrill
    
    # failed: YcbScissors, YcbMustardBottle
    target_obj_name = "YcbBanana" 
    
    # reset simulation with target object
    sim.reset(target_obj_name)
    
    # Initialize point cloud collection list
    collected_data = []
    
    # 获取并保存仿真环境开始时的初始位置
    initial_joints = sim.robot.get_joint_positions()
    print("保存仿真环境初始关节位置")
    
    # 初始化物体高度变量，默认值
    object_height_with_offset = 1.6
    # 初始化物体质心坐标，默认值
    object_centroid_x = -0.02
    object_centroid_y = -0.45

    pause_time = 2.0  # 停顿2秒
    print(f"\n停顿 {pause_time} 秒...")
    for _ in range(int(pause_time * 240)):  # 假设模拟频率为240Hz
        sim.step()
        time.sleep(1/240.)
        
    # ===== 移动到指定位置并获取点云 =====
    print("\n移动到高点观察位置...")
    # 定义高点观察位置和方向
    z_observe_pos = np.array([-0.02, -0.45, 1.9])
    z_observe_orn = p.getQuaternionFromEuler([0, np.radians(-180), 0])  # 向下看
    
    # 解算IK
    ik_solver = DifferentialIKSolver(sim.robot.id, sim.robot.ee_idx, damping=0.05)
    high_point_target_joints = ik_solver.solve(z_observe_pos, z_observe_orn, initial_joints, max_iters=50, tolerance=0.001)
    
    # 生成轨迹
    print("为高点观察位置生成轨迹...")
    # high_point_trajectory = generate_cartesian_trajectory(sim, ik_solver, initial_joints, z_observe_pos, z_observe_orn, steps=100)
    high_point_trajectory = generate_trajectory(initial_joints, high_point_target_joints, steps=100)
    if not high_point_trajectory:
        print("无法生成到高点观察位置的轨迹，跳过高点点云采集")
    else:
        print(f"生成了包含 {len(high_point_trajectory)} 个点的轨迹")
        
        # 重置到初始位置
        for i, joint_idx in enumerate(sim.robot.arm_idx):
            p.resetJointState(sim.robot.id, joint_idx, initial_joints[i])
        
        # 沿轨迹移动机器人到高点
        for joint_target in high_point_trajectory:
            # sim.get_ee_renders()
            sim.robot.position_control(joint_target)
            for _ in range(1):
                sim.step()
                time.sleep(1/240.)
        
        # 在高点观察位置获取点云
        rgb_ee, depth_ee, seg_ee = sim.get_ee_renders()
        camera_pos, camera_R = get_ee_camera_params(sim.robot, config)
        print(f"高点观察位置相机位置:", camera_pos)
        print(f"高点观察位置末端执行器位置:", sim.robot.get_ee_pose()[0])
        
        # 构建点云
        target_mask_id = sim.object.id
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
            
            high_point_pcd = build_object_point_cloud_ee(rgb_ee, depth_ee, seg_ee, target_mask_id, config, camera_pos, camera_R)
            
            # 处理点云
            high_point_pcd = high_point_pcd.voxel_down_sample(voxel_size=0.005)
            high_point_pcd, _ = high_point_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            
            # 存储点云数据
            high_point_cloud_data = {
                'point_cloud': high_point_pcd,
                'camera_position': camera_pos,
                'camera_rotation': camera_R,
                'ee_position': sim.robot.get_ee_pose()[0],
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
            
            # 可视化高点点云
            # print("\n可视化高点点云...")
            # visualize_point_clouds([high_point_cloud_data], show_merged=False)
            
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
    
    # For each viewpoint
    for viewpoint_idx, (target_pos, target_orn) in enumerate(zip(target_positions, target_orientations)):
        print(f"\nMoving to viewpoint {viewpoint_idx + 1}")
        sim.get_ee_renders()
        # Get initial static camera view to setup obstacle tracking
        # rgb_static, depth_static, seg_static = sim.get_static_renders()
        # detections = obstacle_tracker.detect_obstacles(rgb_static, depth_static, seg_static)
        # tracked_positions = obstacle_tracker.update(detections)
        
        # Get current joint positions
        current_joints = sim.robot.get_joint_positions()
        # Save current joint positions
        saved_joints = current_joints.copy()
        
        # Solve IK for target end-effector pose
        ik_solver = DifferentialIKSolver(sim.robot.id, sim.robot.ee_idx, damping=0.05)
        target_joints = ik_solver.solve(target_pos, target_orn, current_joints, max_iters=50, tolerance=0.001)
        
        # Reset to saved start position
        for i, joint_idx in enumerate(sim.robot.arm_idx):
            p.resetJointState(sim.robot.id, joint_idx, saved_joints[i])
        
        choice = 2  # Change this to test different methods
        
        trajectory = []
        if choice == 1:
            print("Generating linear Cartesian trajectory...")
            trajectory = generate_cartesian_trajectory(sim, ik_solver, saved_joints, target_pos, target_orn, steps=100)
        elif choice == 2:
            print("Generating linear joint space trajectory...")
            trajectory = generate_trajectory(saved_joints, target_joints, steps=100)
        
        if not trajectory:
            print(f"Failed to generate trajectory for viewpoint {viewpoint_idx + 1}. Skipping...")
            continue
        
        print(f"Generated trajectory with {len(trajectory)} points")
        
        # Reset to saved start position again before executing trajectory
        for i, joint_idx in enumerate(sim.robot.arm_idx):
            p.resetJointState(sim.robot.id, joint_idx, saved_joints[i])
        
        # Move robot along trajectory to target position
        for joint_target in trajectory:
            # sim.get_ee_renders()
            # Update obstacle tracking
            # rgb_static, depth_static, seg_static = sim.get_static_renders()
            # detections = obstacle_tracker.detect_obstacles(rgb_static, depth_static, seg_static)
            # tracked_positions = obstacle_tracker.update(detections)
            
            # Visualize tracked obstacles
            # bounding_box = obstacle_tracker.visualize_tracking_3d(tracked_positions)
            # if bounding_box:
            #     for debug_line in bounding_box:
            #         p.removeUserDebugItem(debug_line)
            
            # Move robot
            sim.robot.position_control(joint_target)
            for _ in range(1):
                sim.step()
                time.sleep(1/240.)
        
        # Capture point cloud at this viewpoint
        rgb_ee, depth_ee, seg_ee = sim.get_ee_renders()
        camera_pos, camera_R = get_ee_camera_params(sim.robot, config)
        print(f"Viewpoint {viewpoint_idx + 1} camera position:", camera_pos)
        print(f"Viewpoint {viewpoint_idx + 1} end effector position:", sim.robot.get_ee_pose()[0])
        
        # Build point cloud
        target_mask_id = sim.object.id
        print(f"Target object ID: {target_mask_id}")
        
        try:
            if target_mask_id not in np.unique(seg_ee):
                print("Warning: Target object ID not found in segmentation mask.")
                print("Available IDs in segmentation mask:", np.unique(seg_ee))
                
                non_zero_ids = np.unique(seg_ee)[1:] if len(np.unique(seg_ee)) > 1 else []
                if len(non_zero_ids) > 0:
                    target_mask_id = non_zero_ids[0]
                    print(f"Using first non-zero ID instead: {target_mask_id}")
                else:
                    raise ValueError("No valid objects found in segmentation mask")
            
            pcd_ee = build_object_point_cloud_ee(rgb_ee, depth_ee, seg_ee, target_mask_id, config, camera_pos, camera_R)
            
            # Process point cloud
            pcd_ee = pcd_ee.voxel_down_sample(voxel_size=0.005)
            pcd_ee, _ = pcd_ee.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            
            # Store point cloud data
            point_cloud_data = {
                'point_cloud': pcd_ee,
                'camera_position': camera_pos,
                'camera_rotation': camera_R,
                'ee_position': sim.robot.get_ee_pose()[0],
                'timestamp': time.time(),
                'target_object': target_obj_name,
                'viewpoint_idx': viewpoint_idx
            }
            collected_data.append(point_cloud_data)
            print(f"Point cloud collected from viewpoint {viewpoint_idx + 1} with {len(pcd_ee.points)} points.")
            
        except ValueError as e:
            print(f"Error building point cloud for viewpoint {viewpoint_idx + 1}:", e)

    return collected_data

if __name__ == "__main__":
    with open("configs/test_config.yaml", "r") as stream:
        config = yaml.safe_load(stream)
    # Run simulation and collect point clouds
    sim = Simulation(config)
    collected_point_clouds = run_pcd(config)
    print(f"Successfully collected {len(collected_point_clouds)} point clouds.")
    
    # 检查并打印高点点云的z轴最大值点
    for data in collected_point_clouds:
        if data.get('viewpoint_idx') == 'high_point' and 'max_z_point' in data:
            print(f"\n高点观察位置点云的z轴最大值点: {data['max_z_point']}")
    
    # # Visualize the collected point clouds if any were collected
    # if collected_point_clouds:
    #     # First show individual point clouds
    #     print("\nVisualizing individual point clouds...")
    #     visualize_point_clouds(collected_point_clouds, show_merged=False)
        
    #     # Then show merged point cloud
    #     print("\nVisualizing merged point cloud...")
    #     visualize_point_clouds(collected_point_clouds, show_merged=True)
        
        # 执行抓取生成
        print("\n执行抓取生成...")
        run_grasping(config, sim, collected_point_clouds)
        
        # 执行路径规划
        print("\n执行路径规划...")
        run_planning(config, sim)
        
        # 等待用户按下Enter键后关闭模拟
        input("\n按下Enter键关闭模拟...")
        
        # 关闭模拟
        sim.close()