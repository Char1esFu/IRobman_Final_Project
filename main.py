import os
import glob
import yaml
import time
import numpy as np
import argparse
import pybullet as p

from typing import Dict, Any

from pybullet_object_models import ycb_objects  # type:ignore

from src.simulation import Simulation
from src.point_cloud.point_cloud import PointCloudCollector
from src.point_cloud.bounding_box import BoundingBox
from src.path_planning.rrt_star import RRTStarPlanner
from src.path_planning.rrt_star_cartesian import RRTStarCartesianPlanner
from src.obstacle_tracker import ObstacleTracker
from src.grasping.grasping import GraspGeneration, GraspExecution
from src.ik_solver import DifferentialIKSolver
import src.grasping.utils as utils

def run_exp(config: Dict[str, Any]):
    # Example Experiment Runner File
    print("Simulation Start:")
    print(config['world_settings'], config['robot_settings'])
    object_root_path = ycb_objects.getDataPath()
    files = glob.glob(os.path.join(object_root_path, "Ycb*"))
    obj_names = [file.split('/')[-1] for file in files]
    sim = Simulation(config)
    for obj_name in obj_names:
        for tstep in range(10):
            sim.reset(obj_name)
            print((f"Object: {obj_name}, Timestep: {tstep},"
                   f" pose: {sim.get_ground_tuth_position_object}"))
            pos, ori = sim.robot.pos, sim.robot.ori
            print(f"Robot inital pos: {pos} orientation: {ori}")
            l_lim, u_lim = sim.robot.lower_limits, sim.robot.upper_limits
            print(f"Robot Joint Range {l_lim} -> {u_lim}")
            sim.robot.print_joint_infos()
            jpos = sim.robot.get_joint_positions()
            print(f"Robot current Joint Positions: {jpos}")
            jvel = sim.robot.get_joint_velocites()
            print(f"Robot current Joint Velocites: {jvel}")
            ee_pos, ee_ori = sim.robot.get_ee_pose()
            print(f"Robot End Effector Position: {ee_pos}")
            print(f"Robot End Effector Orientation: {ee_ori}")
            for i in range(10000):
                sim.step()
                # for getting renders
                # rgb, depth, seg = sim.get_ee_renders()
                # rgb, depth, seg = sim.get_static_renders()
                obs_position_guess = np.zeros((2, 3))
                print((f"[{i}] Obstacle Position-Diff: "
                       f"{sim.check_obstacle_position(obs_position_guess)}"))
                goal_guess = np.zeros((7,))
                print((f"[{i}] Goal Obj Pos-Diff: "
                       f"{sim.check_goal_obj_pos(goal_guess)}"))
                print(f"[{i}] Goal Satisfied: {sim.check_goal()}")
    sim.close()


def collect_point_clouds(config: Dict[str, Any], target_obj_name=None):
    """
    运行点云采集流程
    
    参数:
    config: 配置字典
    target_obj_name: 目标物体名称，如果为None则随机选择
    
    返回:
    sim: 仿真环境对象
    collector: 点云收集器对象
    collected_point_clouds: 收集的点云数据
    """
    print("步骤1: 开始点云采集...")
    
    # 初始化模拟环境
    sim = Simulation(config)
    
    # 创建点云收集器
    collector = PointCloudCollector(config, sim)
    
    # 如果没有指定目标物体，默认使用"YcbBanana"
    if target_obj_name is None:
        target_obj_name = "YcbBanana"
    
    # 收集点云
    collected_point_clouds = collector.collect_point_clouds(target_obj_name)
    print(f"成功收集了 {len(collected_point_clouds)} 个点云。")
    
    # 检查并打印高点点云的z轴最大值点
    for data in collected_point_clouds:
        if data.get('viewpoint_idx') == 'high_point' and 'max_z_point' in data:
            print(f"\n高点观察位置点云的z轴最大值点: {data['max_z_point']}")
    
    return sim, collector, collected_point_clouds


def compute_bounding_box(sim, collector, point_clouds, visualize_cloud=True):
    """
    计算和可视化点云边界框
    
    参数:
    sim: 仿真环境对象
    collector: 点云收集器对象
    point_clouds: 收集的点云数据
    visualize_cloud: 是否可视化点云
    
    返回:
    bbox: 计算的边界框对象
    """
    print("\n步骤2: 计算和可视化边界框...")
    
    # 可视化收集的点云
    if visualize_cloud and point_clouds:
        # 显示单独的点云
        print("\n可视化单独点云...")
        collector.visualize_point_clouds(point_clouds, show_merged=False)
    
    # 合并点云
    print("\n合并点云...")
    merged_cloud = collector.merge_point_clouds(point_clouds)
    
    # 可视化合并点云
    if visualize_cloud and merged_cloud is not None:
        print("\n可视化合并点云...")
        # 创建一个只包含合并点云的列表
        merged_cloud_data = [{
            'point_cloud': merged_cloud,
            'camera_position': np.array([0, 0, 0]),  # 占位符
            'camera_rotation': np.eye(3)  # 占位符
        }]
        collector.visualize_point_clouds(merged_cloud_data, show_merged=False)
    
    # 计算边界框
    print("\n计算边界框...")
    bbox = BoundingBox(merged_cloud)
    bbox.compute_obb()
    
    # 可视化边界框
    print("\n可视化边界框...")
    bbox.visualize_in_pybullet(color=(0, 1, 1), line_width=3)
    
    # 可视化中心点
    centroid_id = bbox.add_centroid_visualization(radius=0.02)
    
    # 可视化主轴
    axis_lines = bbox.add_axes_visualization(length=0.15)
    
    # 打印边界框信息
    print(f"\n边界框信息:")
    print(f"物体高度: {bbox.get_height():.4f}米")
    print(f"边界框尺寸: {bbox.get_dimensions()}")
    center = bbox.get_center()
    print(f"质心坐标: ({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f})")
    
    return bbox


def execute_grasping(sim, bbox, point_clouds, visualize=True):
    """
    执行抓取规划和执行
    
    参数:
    sim: 仿真环境对象
    bbox: 边界框对象
    point_clouds: 收集的点云数据
    visualize: 是否可视化抓取过程
    
    返回:
    success: 抓取是否成功
    grasp_executor: 抓取执行器（如果抓取成功）
    """
    print("\n步骤3: 抓取规划和执行...")
    
    # 合并点云
    print("\n准备合并点云...")
    merged_pcd = None
    for data in point_clouds:
        if 'point_cloud' in data and data['point_cloud'] is not None:
            if merged_pcd is None:
                merged_pcd = data['point_cloud']
            else:
                merged_pcd += data['point_cloud']
    
    if merged_pcd is None:
        print("错误：无法合并点云，终止抓取")
        return False
    
    # 获取边界框信息
    center = bbox.get_center()
    rotation_matrix = bbox.get_rotation_matrix()
    min_point, max_point = bbox.get_aabb()
    obb_corners = bbox.get_corners()
    
    # 获取旋转边界框内的坐标
    points_rotated = np.dot(np.asarray(merged_pcd.points) - center, rotation_matrix)
    min_point_rotated = np.min(points_rotated, axis=0)
    max_point_rotated = np.max(points_rotated, axis=0)
    
    print(f"\n边界框信息:")
    print(f"质心坐标: {center}")
    print(f"旋转坐标系中最小点: {min_point_rotated}")
    print(f"旋转坐标系中最大点: {max_point_rotated}")
    
    grasp_generator = GraspGeneration()
    
    # 生成抓取候选
    print("\n生成抓取候选...")
    sampled_grasps = grasp_generator.sample_grasps(
        center, 
        num_grasps=100, 
        sim=sim,
        rotation_matrix=rotation_matrix,
        min_point_rotated=min_point_rotated,
        max_point_rotated=max_point_rotated,
        center_rotated=center
    )
    
    # 为每个抓取创建网格
    all_grasp_meshes = []
    for grasp in sampled_grasps:
        R, grasp_center = grasp
        all_grasp_meshes.append(utils.create_grasp_mesh(center_point=grasp_center, rotation_matrix=R))
    
    # 评估抓取质量
    print("\n评估抓取质量...")
    
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
                print(f"找到更好的抓取，质量: {grasp_quality:.3f}")
    
    if best_grasp is None:
        print("未找到有效抓取！")
        return False, None
    
    print(f"\n找到最佳抓取，质量分数: {highest_quality:.4f}")
    
    # 初始化抓取执行器
    grasp_executor = GraspExecution(sim)
    
    # 计算抓取姿态
    pose1_pos, pose1_orn, pose2_pos, pose2_orn = grasp_executor.compute_grasp_poses(best_grasp)
    
    # 可视化抓取姿态
    if visualize:
        grasp_generator.visualize_grasp_poses(
            pose1_pos, pose1_orn, pose2_pos, pose2_orn, axis_length=0.1
        )
    
    # 执行抓取
    print("\n开始执行抓取...")
    success = grasp_executor.execute_grasp(best_grasp)
    
    if success:
        print("\n抓取成功！")
    else:
        print("\n抓取失败...")
    
    return success, grasp_executor if success else None


def execute_path_planning(sim, grasp_executor, planning_type='joint', visualize=True):
    """
    执行路径规划
    
    参数:
    sim: 仿真环境对象
    grasp_executor: 抓取执行器对象
    planning_type: 规划类型 ('joint' 或 'cartesian')
    visualize: 是否可视化规划过程
    
    返回:
    success: 规划是否成功
    """
    print(f"\n步骤4: {'笛卡尔空间' if planning_type == 'cartesian' else '关节空间'}路径规划...")
    
    # 获取机器人当前状态（抓取后的位置）作为起点
    robot = sim.robot
    joint_indices = robot.arm_idx
    ee_link_index = robot.ee_idx
    
    # 获取关节限制
    lower_limits = robot.lower_limits
    upper_limits = robot.upper_limits
    
    # 获取当前关节位置
    start_joint_pos = robot.get_joint_positions()
    
    # 获取目标托盘位置
    min_lim, max_lim = sim.goal._get_goal_lims()
    goal_pos = np.array([
        (min_lim[0] + max_lim[0])/2 - 0.1,
        (min_lim[1] + max_lim[1])/2 - 0.1,
        max_lim[2] + 0.2
    ])
    print(f"托盘目标位置: {goal_pos}")
    
    # 目标位置和方向（托盘上方，合理的距离）
    tray_approach_pos = goal_pos.copy()  # 托盘上方的位置
    tray_orn = p.getQuaternionFromEuler([0, np.pi, 0])  # 竖直向下
    
    # 在PyBullet中可视化托盘目标位置
    if visualize:
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
    
    # 创建障碍物跟踪器
    obstacle_tracker = ObstacleTracker(n_obstacles=2, exp_settings=config)
    
    # 使用静态相机获取障碍物位置
    rgb_static, depth_static, seg_static = sim.get_static_renders()
    detections = obstacle_tracker.detect_obstacles(rgb_static, depth_static, seg_static)
    tracked_positions = obstacle_tracker.update(detections)
    
    # 可视化障碍物边界框（如果需要）
    if visualize:
        bounding_box_ids = obstacle_tracker.visualize_tracking_3d(tracked_positions)
        print(f"检测到 {len(tracked_positions)} 个障碍物")
    
    # 创建适当的规划器
    if planning_type == 'cartesian':
        # 创建笛卡尔空间规划器
        planner = RRTStarCartesianPlanner(
            robot_id=robot.id,
            joint_indices=joint_indices,
            lower_limits=lower_limits,
            upper_limits=upper_limits,
            ee_link_index=ee_link_index,
            obstacle_tracker=obstacle_tracker,
            max_iterations=1000,
            step_size=0.05,
            goal_sample_rate=0.1,
            search_radius=0.1,
            goal_threshold=0.03
        )
        
        # 在笛卡尔空间中规划
        print(f"\n使用笛卡尔空间RRT*进行规划...")
        path, cost = planner.plan(start_joint_pos, tray_approach_pos, tray_orn)
        
    else:
        # 创建关节空间规划器
        planner = RRTStarPlanner(
            robot_id=robot.id,
            joint_indices=joint_indices,
            lower_limits=lower_limits,
            upper_limits=upper_limits,
            ee_link_index=ee_link_index,
            obstacle_tracker=obstacle_tracker,
            max_iterations=1000,
            step_size=0.2,
            goal_sample_rate=0.05,
            search_radius=0.5,
            goal_threshold=0.1
        )
        
        # 尝试将目标笛卡尔位置转换为关节空间
        ik_solver = DifferentialIKSolver(robot.id, ee_link_index, damping=0.05)
        
        try:
            goal_joint_pos = ik_solver.solve(tray_approach_pos, tray_orn, start_joint_pos, max_iters=50, tolerance=0.001)
            print(f"目标位置IK解: {goal_joint_pos}")
        except:
            print("无法为目标位置找到IK解")
            return False
        
        # 在关节空间中规划
        print(f"\n使用关节空间RRT*进行规划...")
        path, cost = planner.plan(start_joint_pos, goal_joint_pos)
    
    if not path:
        print("未找到路径")
        return False
    
    print(f"找到路径！代价: {cost:.4f}，路径点数: {len(path)}")
    
    # 生成平滑轨迹
    print("\n生成平滑轨迹...")
    smooth_path = planner.generate_smooth_trajectory(path, smoothing_steps=20)
    
    # 执行轨迹
    print("\n执行轨迹...")
    for joint_pos in smooth_path:
        # 设置关节位置
        for i, idx in enumerate(joint_indices):
            p.setJointMotorControl2(robot.id, idx, p.POSITION_CONTROL, joint_pos[i])
        
        # 更新仿真
        sim.step()
        time.sleep(0.01)
    
    print("\n路径执行完成")
    return True


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='机器人点云采集和边界框计算')
    # All objects: 
    # Low objects: YcbBanana, YcbFoamBrick, YcbHammer, YcbMediumClamp, YcbPear, YcbScissors, YcbStrawberry, YcbTennisBall, 
    # Medium objects: YcbGelatinBox, YcbMasterChefCan, YcbPottedMeatCan, YcbTomatoSoupCan
    # High objects: YcbCrackerBox, YcbMustardBottle, 
    # Unstable objects: YcbChipsCan, YcbPowerDrill
    parser.add_argument('--object', type=str, default="YcbCrackerBox",
                        help='目标物体名称')
    parser.add_argument('--no-vis', action='store_true',
                        help='禁用点云可视化')
    parser.add_argument('--no-grasp', action='store_true',
                        help='禁用抓取执行')
    parser.add_argument('--no-planning', action='store_true',
                        help='禁用路径规划')
    parser.add_argument('--planning-type', type=str, choices=['joint', 'cartesian'], default='joint',
                        help='选择路径规划类型: joint(关节空间) 或 cartesian(笛卡尔空间)')
    
    args = parser.parse_args()
    
    # 加载配置文件
    config_path = "configs/test_config.yaml"
    with open(config_path, "r") as stream:
        try:
            config = yaml.safe_load(stream)
            print(f"加载配置文件: {config_path}")
        except yaml.YAMLError as exc:
            print(f"配置文件加载错误: {exc}")
    
    # 变量初始化
    sim = None
    collector = None
    point_clouds = None
    bbox = None
    grasp_success = False
    grasp_executor = None
    
    # 步骤1: 收集点云
    sim, collector, point_clouds = collect_point_clouds(config, args.object)
    
    # 步骤2: 计算和可视化边界框
    if point_clouds:
        bbox = compute_bounding_box(sim, collector, point_clouds, not args.no_vis)
        
        # 步骤3: 执行抓取（除非--no-grasp标志被设置）
        if not args.no_grasp and bbox is not None:
            grasp_success, grasp_executor = execute_grasping(sim, bbox, point_clouds, True)
            
            # 步骤4: 执行路径规划（如果抓取成功且未禁用规划）
            if grasp_success and not args.no_planning and grasp_executor is not None:
                planning_success = execute_path_planning(
                    sim, 
                    grasp_executor, 
                    planning_type=args.planning_type,
                    visualize=True
                )
                if planning_success:
                    print(f"\n{'笛卡尔空间' if args.planning_type == 'cartesian' else '关节空间'}路径规划执行成功！")
                else:
                    print(f"\n{'笛卡尔空间' if args.planning_type == 'cartesian' else '关节空间'}路径规划执行失败。")
    
    input("\n按下Enter键关闭模拟...")
    
    # 关闭模拟
    if sim is not None:
        sim.close()
