import numpy as np
import pybullet as p
import time
from typing import Optional, Tuple, List, Any, Dict

from src.path_planning.rrt_star import RRTStarPlanner
from src.path_planning.rrt_star_cartesian import RRTStarCartesianPlanner
from src.obstacle_tracker import ObstacleTracker
from src.ik_solver import DifferentialIKSolver

class PlanningExecutor:
    """
    路径规划执行器，负责执行机器人从抓取位置到目标位置的路径规划
    可以根据指定的规划类型选择使用关节空间或笛卡尔空间的规划器
    """
    
    def __init__(self, sim, config: Dict[str, Any]):
        """
        初始化路径规划执行器
        
        参数:
        sim: 仿真环境对象
        config: 配置参数字典
        """
        self.sim = sim
        self.config = config
        self.robot = sim.robot
        self.obstacle_tracker = ObstacleTracker(n_obstacles=2, exp_settings=config)
        
        # 初始化IK求解器
        self.ik_solver = DifferentialIKSolver(
            self.robot.id, 
            self.robot.ee_idx, 
            damping=0.05
        )
    
    def execute_planning(self, grasp_executor, planning_type='joint', visualize=True) -> bool:
        """
        执行路径规划
        
        参数:
        grasp_executor: 抓取执行器对象
        planning_type: 规划类型 ('joint' 或 'cartesian')
        visualize: 是否可视化规划过程
        
        返回:
        success: 规划是否成功
        """
        print(f"\n步骤4: {'笛卡尔空间' if planning_type == 'cartesian' else '关节空间'}路径规划...")
        
        # 获取机器人当前状态（抓取后的位置）作为起点
        joint_indices = self.robot.arm_idx
        ee_link_index = self.robot.ee_idx
        
        # 获取关节限制
        lower_limits = self.robot.lower_limits
        upper_limits = self.robot.upper_limits
        
        # 获取当前关节位置
        start_joint_pos = self.robot.get_joint_positions()
        
        # 获取目标托盘位置
        min_lim, max_lim = self.sim.goal._get_goal_lims()
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
            self._visualize_goal_position(goal_pos)
        
        # 使用静态相机获取障碍物位置
        rgb_static, depth_static, seg_static = self.sim.get_static_renders()
        detections = self.obstacle_tracker.detect_obstacles(rgb_static, depth_static, seg_static)
        tracked_positions = self.obstacle_tracker.update(detections)
        
        # 可视化障碍物边界框（如果需要）
        if visualize:
            bounding_box_ids = self.obstacle_tracker.visualize_tracking_3d(tracked_positions)
            print(f"检测到 {len(tracked_positions)} 个障碍物")
        
        # 根据规划类型选择和使用适当的规划器
        if planning_type == 'cartesian':
            # 使用笛卡尔空间规划
            path, cost = self._execute_cartesian_planning(
                start_joint_pos, tray_approach_pos, tray_orn, visualize
            )
        else:
            # 使用关节空间规划
            path, cost = self._execute_joint_planning(
                start_joint_pos, tray_approach_pos, tray_orn, visualize
            )
        
        if not path:
            print("未找到路径")
            return False
        
        print(f"找到路径！代价: {cost:.4f}，路径点数: {len(path)}")
        
        # 获取使用的规划器
        planner = self._get_planner(planning_type)
        
        # 可视化轨迹
        if visualize and planner:
            self._visualize_path(planner, path)
        
        # 生成平滑轨迹
        print("\n生成平滑轨迹...")
        smooth_path = planner.generate_smooth_trajectory(path, smoothing_steps=20)
        
        # 执行轨迹
        print("\n执行轨迹...")
        self._execute_trajectory(joint_indices, smooth_path)
        
        print("\n路径执行完成")
        
        # 放下物体
        print("\n放下物体...")
        self._release_object()
        
        print("爪子已打开，物体已放置到托盘位置")
        
        return True
    
    def _execute_cartesian_planning(self, start_joint_pos, goal_pos, goal_orn, visualize):
        """执行笛卡尔空间规划"""
        # 创建笛卡尔空间规划器
        planner = RRTStarCartesianPlanner(
            robot_id=self.robot.id,
            joint_indices=self.robot.arm_idx,
            lower_limits=self.robot.lower_limits,
            upper_limits=self.robot.upper_limits,
            ee_link_index=self.robot.ee_idx,
            obstacle_tracker=self.obstacle_tracker,
            max_iterations=1000,
            step_size=0.05,
            goal_sample_rate=0.1,
            search_radius=0.1,
            goal_threshold=0.03
        )
        
        # 在笛卡尔空间中规划
        print(f"\n使用笛卡尔空间RRT*进行规划...")
        return planner.plan(start_joint_pos, goal_pos, goal_orn)
    
    def _execute_joint_planning(self, start_joint_pos, goal_pos, goal_orn, visualize):
        """执行关节空间规划"""
        # 创建关节空间规划器
        planner = RRTStarPlanner(
            robot_id=self.robot.id,
            joint_indices=self.robot.arm_idx,
            lower_limits=self.robot.lower_limits,
            upper_limits=self.robot.upper_limits,
            ee_link_index=self.robot.ee_idx,
            obstacle_tracker=self.obstacle_tracker,
            max_iterations=1000,
            step_size=0.2,
            goal_sample_rate=0.05,
            search_radius=0.5,
            goal_threshold=0.1
        )
        
        # 尝试将目标笛卡尔位置转换为关节空间
        try:
            goal_joint_pos = self.ik_solver.solve(
                goal_pos, goal_orn, start_joint_pos, max_iters=50, tolerance=0.001
            )
            print(f"目标位置IK解: {goal_joint_pos}")
        except Exception as e:
            print(f"无法为目标位置找到IK解: {e}")
            return None, 0
        
        # 在关节空间中规划
        print(f"\n使用关节空间RRT*进行规划...")
        return planner.plan(start_joint_pos, goal_joint_pos)
    
    def _get_planner(self, planning_type):
        """根据规划类型获取对应的规划器实例"""
        if planning_type == 'cartesian':
            return RRTStarCartesianPlanner(
                robot_id=self.robot.id,
                joint_indices=self.robot.arm_idx,
                lower_limits=self.robot.lower_limits,
                upper_limits=self.robot.upper_limits,
                ee_link_index=self.robot.ee_idx,
                obstacle_tracker=self.obstacle_tracker
            )
        elif planning_type == 'joint':
            return RRTStarPlanner(
                robot_id=self.robot.id,
                joint_indices=self.robot.arm_idx,
                lower_limits=self.robot.lower_limits,
                upper_limits=self.robot.upper_limits,
                ee_link_index=self.robot.ee_idx,
                obstacle_tracker=self.obstacle_tracker
            )
    
    def _visualize_goal_position(self, goal_pos):
        """可视化目标位置"""
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
    
    def _visualize_path(self, planner, path):
        """可视化规划路径"""
        # 清除之前的可视化
        planner.clear_visualization()
        
        # 可视化路径
        for i in range(len(path) - 1):
            start_ee, _ = planner._get_current_ee_pose(path[i])
            end_ee, _ = planner._get_current_ee_pose(path[i+1])
            
            p.addUserDebugLine(
                start_ee, end_ee, [0, 0, 1], 3, 0)
    
    def _execute_trajectory(self, joint_indices, trajectory):
        """执行轨迹"""
        for joint_pos in trajectory:
            # 设置关节位置
            for i, idx in enumerate(joint_indices):
                p.setJointMotorControl2(self.robot.id, idx, p.POSITION_CONTROL, joint_pos[i])
            
            # 更新仿真
            self.sim.step()
            time.sleep(0.01)
    
    def _release_object(self):
        """放下物体"""
        # 打开爪子
        open_gripper_width = 0.04  # 打开爪子的宽度
        p.setJointMotorControlArray(
            self.robot.id,
            jointIndices=self.robot.gripper_idx,
            controlMode=p.POSITION_CONTROL,
            targetPositions=[open_gripper_width, open_gripper_width]
        )
        
        # 等待爪子打开
        for _ in range(int(1.0 * 240)):  # 等待1秒
            self.sim.step()
            time.sleep(1/240.)
