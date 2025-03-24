import numpy as np
import pybullet as p
import time
from typing import Optional, Tuple, List, Any, Dict

from src.path_planning.rrt_star import RRTStarPlanner
from src.path_planning.rrt_star_cartesian import RRTStarCartesianPlanner
from src.path_planning.potential_field import PotentialFieldPlanner  
from src.obstacle_tracker import ObstacleTracker
from src.ik_solver import DifferentialIKSolver
from src.path_planning.simple_planning import SimpleTrajectoryPlanner
class PlanningExecutor:
    """
    Path planning executor, responsible for executing robot path planning from grasp position to target position.
    Can choose between joint space or Cartesian space planners based on the specified planning type.
    """
    
    def __init__(self, sim, config: Dict[str, Any]):
        """
        Initialize path planning executor
        
        Parameters:
        sim: Simulation environment object
        config: Configuration parameter dictionary
        """
        self.sim = sim
        self.config = config
        self.robot = sim.robot
        self.obstacle_tracker = ObstacleTracker(n_obstacles=2, exp_settings=config)
        
        # Initialize IK solver
        self.ik_solver = DifferentialIKSolver(
            self.robot.id, 
            self.robot.ee_idx, 
            damping=0.05
        )
    
    def execute_planning(self, grasp_executor, planning_type='joint', visualize=True, 
                         movement_speed_factor=1.0, enable_replan=False, replan_steps=10 ,method= "Hard_Code") -> bool:
        """
        Execute path planning
        
        Parameters:
        grasp_executor: Grasp executor object
        planning_type: Planning type ('joint' or 'cartesian')
        visualize: Whether to visualize the planning process
        movement_speed_factor: Speed factor for trajectory execution (lower = faster, higher = slower)
        enable_replan: Whether to enable dynamic replanning
        replan_steps: Number of steps to execute before replanning (if enable_replan is True)
        
        Returns:
        success: Whether planning was successful
        """
        print(f"\nStep 4: {'Cartesian space' if planning_type == 'cartesian' else 'Joint space'} path planning...")
        print(f"动态重规划: {'启用' if enable_replan else '禁用'}")
        
        # Get robot's current state (position after grasping) as starting point
        joint_indices = self.robot.arm_idx
        ee_link_index = self.robot.ee_idx
        

        if(method == "Hard_Code"):   
            start_pos = np.array([
                0.1,
                0.1,
                2.5
            ])
            start_orn = p.getQuaternionFromEuler([0, 0, 0])  # Vertically downward

            # if visualize:
            #     self._visualize_goal_position(start_pos)

            current_joint_pos = self.robot.get_joint_positions()

            start_joint_pos = self.ik_solver.solve(start_pos, start_orn, current_joint_pos, max_iters=50, tolerance=0.001)
            start_joint_pos[0] = 0.9

            Path_start = SimpleTrajectoryPlanner.generate_joint_trajectory(current_joint_pos, start_joint_pos, steps=100)
            for joint_target in Path_start:
                    self.sim.robot.position_control(joint_target)
                    for _ in range(10):
                        self.sim.step()
                        time.sleep(1/240.)
            current_joint_pos = self.robot.get_joint_positions()

            # start_joint_pos[0] = 0.9
            # Path_start = SimpleTrajectoryPlanner.generate_joint_trajectory(current_joint_pos, start_joint_pos, steps=100)
            # for joint_target in Path_start:
            #         self.sim.robot.position_control(joint_target)
            #         for _ in range(10):
            #             self.sim.step()
            #             time.sleep(1/240.)
            # current_joint_pos = self.robot.get_joint_positions()

            start_joint_pos = [0.9, 0.4099985502678034, 0.0026639666712291836, -0.67143263171620764, 0.0, 3.14*0.8, 2.9671]
            Path_start = SimpleTrajectoryPlanner.generate_joint_trajectory(current_joint_pos, start_joint_pos, steps=100)
            for joint_target in Path_start:
                    self.sim.robot.position_control(joint_target)
                    for _ in range(10):
                        self.sim.step()
                        time.sleep(1/240.)
            current_joint_pos = self.robot.get_joint_positions()

            pos , ori = self.sim.robot.get_ee_pose()
            # print("==================================================", pos , ori)

            ball_is_far = False
            while(not ball_is_far):
                rgb_static, depth_static, seg_static = self.sim.get_static_renders()
                detections = self.obstacle_tracker.detect_obstacles(rgb_static, depth_static, seg_static)
                tracked_positions = self.obstacle_tracker.update(detections)
                ball_is_far = self.obstacle_tracker.is_away()
                for _ in range(1):
                    self.sim.step()
                    time.sleep(1/240.)

   
            start_joint_pos = [0.9, 0.6099985502678034, 0.0026639666712291836, -0.47143263171620764, 0.0, 3.14*0.8, 2.9671]
            Path_start = SimpleTrajectoryPlanner.generate_joint_trajectory(current_joint_pos, start_joint_pos, steps=100)
            for joint_target in Path_start:
                    self.sim.robot.position_control(joint_target)
                    for _ in range(1):
                        self.sim.step()
                        time.sleep(1/240.)
            # 更新当前位置
            current_joint_pos = self.robot.get_joint_positions()

                
            print("\n轨迹执行完成, \n目标位置到达!")
            # Release object
            print("\nReleasing object...")
            self._release_object()
            
            print("Gripper opened, object placed at tray position")

        elif(method == "RRT*_Plan"):
            # Get target tray position
            min_lim, max_lim = self.sim.goal._get_goal_lims()
            goal_pos = np.array([
                (min_lim[0] + max_lim[0])/2 - 0.1,
                (min_lim[1] + max_lim[1])/2 - 0.1,
                max_lim[2] + 0.2
            ])
            goal_orn = p.getQuaternionFromEuler([0, np.pi, 0])  # Vertically downward
            
            # Visualize tray target position in PyBullet
            if visualize:
                self._visualize_goal_position(goal_pos)
            
            # 计算目标关节位置 (仅在关节空间规划中需要)
            if planning_type == 'joint':
                goal_joint_pos = self.ik_solver.solve(
                    goal_pos, goal_orn, self.robot.get_joint_positions(), max_iters=50, tolerance=0.001
                )
            
            # 如果启用了重规划，我们将跟踪当前位置和目标位置
            if enable_replan:
                # 初始创建规划器 (后续会在循环中重用)
                if planning_type == 'cartesian':
                    planner = RRTStarCartesianPlanner(
                        robot_id=self.robot.id,
                        joint_indices=self.robot.arm_idx,
                        lower_limits=self.robot.lower_limits,
                        upper_limits=self.robot.upper_limits,
                        ee_link_index=self.robot.ee_idx,
                        obstacle_tracker=self.obstacle_tracker,
                        max_iterations=500,  # 为重规划减少迭代次数以加快速度
                        step_size=0.05,
                        goal_sample_rate=0.1,
                        search_radius=0.1,
                        goal_threshold=0.03
                    )
                else:  # joint space planning
                    planner = RRTStarPlanner(
                        robot=self.robot,
                        obstacle_tracker=self.obstacle_tracker,
                        max_iterations=2000,
                        step_size=0.1,
                        goal_sample_rate=0.05,
                        search_radius= 0.5,
                        goal_threshold=0.05
                    )
                
                # 主循环：执行路径、监测障碍物和重规划
                print("\n开始执行带有动态重规划的轨迹...")
                
                # 调整执行速度参数
                steps = max(1, int(10 * movement_speed_factor))
                delay = (1/240.0) * movement_speed_factor
                
                current_joint_pos = self.robot.get_joint_positions()
                goal_reached = False
                
                while not goal_reached:
                    # 更新障碍物位置
                    rgb_static, depth_static, seg_static = self.sim.get_static_renders()
                    detections = self.obstacle_tracker.detect_obstacles(rgb_static, depth_static, seg_static)
                    tracked_positions = self.obstacle_tracker.update(detections)
                    
                    # # 可视化障碍物边界框
                    # if visualize:
                    #     self.obstacle_tracker.visualize_tracking_3d(tracked_positions)
                    #     print(f"检测到 {len(tracked_positions)} 个障碍物")
                    
                    # 从当前位置规划到目标
                    print("\n重新规划路径...")
                    if planning_type == 'cartesian':
                        path, cost = planner.plan(current_joint_pos, goal_pos, goal_orn)
                    else:  # joint space planning
                        path, cost = planner.plan(current_joint_pos, goal_joint_pos)
                    
                    if not path:
                        print("无法找到有效路径，尝试再次规划...")
                        time.sleep(0.5)  # 稍等一下再尝试
                        continue
                    
                    print(f"找到路径! 代价: {cost:.4f}, 路径点数量: {len(path)}")
                    
                    # 可视化轨迹
                    if visualize and planner:
                        self._visualize_path(planner, path)
                    
                    # 生成平滑轨迹
                    smooth_path = planner.generate_smooth_trajectory(path, smoothing_steps=5)  # 减少平滑步数以提高响应速度
                    
                    # 只执行轨迹的一部分，然后重新规划
                    subpath = smooth_path[:min(replan_steps, len(smooth_path))]
                    
                    # 执行子轨迹
                    for joint_pos in subpath:
                        # 设置关节位置
                        for i, idx in enumerate(joint_indices):
                            p.setJointMotorControl2(self.robot.id, idx, p.POSITION_CONTROL, joint_pos[i])
                        
                        # 执行多个仿真步骤
                        for _ in range(steps):
                            self.sim.step()
                            time.sleep(delay)
                        
                        # 更新当前位置
                        current_joint_pos = self.robot.get_joint_positions()
                    
                    # 检查是否到达目标
                    if planning_type == 'joint':
                        dist_to_goal = np.linalg.norm(np.array(current_joint_pos) - np.array(goal_joint_pos))
                        goal_reached = dist_to_goal < planner.goal_threshold
                    else:  # cartesian space
                        current_ee_pos, _ = self.robot.get_ee_pose()
                        dist_to_goal = np.linalg.norm(np.array(current_ee_pos) - np.array(goal_pos))
                        goal_reached = dist_to_goal < 0.03  # 厘米级精度
                    
                    if goal_reached:
                        print("\n目标位置到达!")
                
                print("\n轨迹执行完成")
                
            else:
                # 不使用重规划的情况下，使用原始方法执行一次性规划
                # Use static camera to get obstacle positions
                rgb_static, depth_static, seg_static = self.sim.get_static_renders()
                detections = self.obstacle_tracker.detect_obstacles(rgb_static, depth_static, seg_static)
                tracked_positions = self.obstacle_tracker.update(detections)
                
                # Visualize obstacle bounding boxes (if needed)
                if visualize:
                    bounding_box_ids = self.obstacle_tracker.visualize_tracking_3d(tracked_positions)
                    print(f"Detected {len(tracked_positions)} obstacles")
                
                # 获取当前关节位置
                start_joint_pos = self.robot.get_joint_positions()
                
                # Choose and use appropriate planner based on planning type
                if planning_type == 'cartesian':
                    # Use Cartesian space planning
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
                    path, cost = planner.plan(start_joint_pos, goal_pos, goal_orn)
                elif planning_type == 'joint':
                    # Use joint space planning
                    planner = RRTStarPlanner(
                        robot=self.robot,
                        obstacle_tracker=self.obstacle_tracker,
                        max_iterations=2000,
                        step_size=0.1,
                        goal_sample_rate=0.05,
                        search_radius= 0.5,
                        goal_threshold=0.05
                    )
                    goal_joint_pos = self.ik_solver.solve(
                        goal_pos, goal_orn, start_joint_pos, max_iters=50, tolerance=0.001
                    )
                    path, cost = planner.plan(start_joint_pos, goal_joint_pos)
                
                if not path:
                    print("No path found")
                    return False
                
                print(f"Path found! Cost: {cost:.4f}, Number of path points: {len(path)}")
                
                # Visualize trajectory
                if visualize and planner:
                    self._visualize_path(planner, path)
                
                # Generate smooth trajectory
                print("\nGenerating smooth trajectory...")
                smooth_path = planner.generate_smooth_trajectory(path, smoothing_steps=20)
                
                # Execute trajectory
                print("\nExecuting trajectory...")
                # 调整步数和延迟基于速度因子
                steps = int(10 * movement_speed_factor)  # 默认5步，乘以速度因子
                delay = (1/240.0) * movement_speed_factor  # 默认延迟，乘以速度因子
                self._execute_trajectory(joint_indices, smooth_path, steps=steps, delay=delay)
                
                print("\nPath execution completed")

            # 更新当前位置
            current_joint_pos = self.robot.get_joint_positions()
            
            # 检查是否到达目标
            if planning_type == 'joint':
                dist_to_goal = np.linalg.norm(np.array(current_joint_pos) - np.array(goal_joint_pos))
                goal_reached = dist_to_goal < planner.goal_threshold
            else:  # cartesian space
                current_ee_pos, _ = self.robot.get_ee_pose()
                dist_to_goal = np.linalg.norm(np.array(current_ee_pos) - np.array(goal_pos))
                goal_reached = dist_to_goal < 0.03  # 厘米级精度
            
            if goal_reached:
                print("\n目标位置到达!")
                
            print("\n轨迹执行完成")
            # Release object
            print("\nReleasing object...")
            self._release_object()
            
            print("Gripper opened, object placed at tray position")

        elif(method == "Potential_Plan"):
            movement_speed_factor=1.0
            # start_pos = np.array([
            #     0.1,
            #     0.1,
            #     2.5
            # ])
            # start_orn = p.getQuaternionFromEuler([0, 0, 0])  # Vertically downward

            # # if visualize:
            # #     self._visualize_goal_position(start_pos)

            # current_joint_pos = self.robot.get_joint_positions()

            # start_joint_pos = self.ik_solver.solve(start_pos, start_orn, current_joint_pos, max_iters=50, tolerance=0.001)
            # start_joint_pos[0] = 0.9

            # Path_start = SimpleTrajectoryPlanner.generate_joint_trajectory(current_joint_pos, start_joint_pos, steps=100)
            # for joint_target in Path_start:
            #         self.sim.robot.position_control(joint_target)
            #         for _ in range(10):
            #             self.sim.step()
            #             time.sleep(1/240.)
            # current_joint_pos = self.robot.get_joint_positions()
            # ----------------- 1) 获取目标信息 -----------------
            goal_pos = np.array([0.45593274 ,0.55745936, 2.14598578])
            # 可视化目标位置（如需要）
            if visualize:
                self._visualize_goal_position(goal_pos)
            
        
            goal_joint_pos = [0.9, 0.4099985502678034, 0.0026639666712291836, -0.67143263171620764, 0.0, 3.14*0.8, 2.9671]
            
            # ----------------- 2) 初始化势场规划器 -----------------
            print("\n初始化势场规划器用于实时避障...")
            
            pf_planner = PotentialFieldPlanner(
                robot_id=self.robot.id,
                joint_indices=self.robot.arm_idx,
                lower_limits=self.robot.lower_limits,
                upper_limits=self.robot.upper_limits,
                ee_link_index=self.robot.ee_idx,
                obstacle_tracker=self.obstacle_tracker,
                max_iterations=200,       # 每次规划迭代次数不需要太多
                step_size=0.01,           # 势场下降步长
                d0=0.5,                  # 排斥势生效距离
                K_att=1.0,                # 吸引势增益
                K_rep=100.0,                # 排斥势增益（加大以更好避障）
                goal_threshold=0.05,      # 到达目标的阈值
                collision_check_step=0.2,
                reference_path_weight=0.7  # 全局路径的引力权重
            )

            # ----------------- 3) 动态重规划主循环 -----------------
            print("\n开始执行基于 Potential Field 的动态重规划...")
            steps = max(1, int(10 * movement_speed_factor))
            delay = (1 / 240.0) * movement_speed_factor
            
            current_joint_pos = self.robot.get_joint_positions()
            goal_reached = False

            while not goal_reached:
                # (a) 获取当前环境的图像并更新障碍物
                rgb_static, depth_static, seg_static = self.sim.get_static_renders()
                detections = self.obstacle_tracker.detect_obstacles(rgb_static, depth_static, seg_static)
                tracked_positions = self.obstacle_tracker.update(detections)

                # (b) 使用势场法重新规划，考虑全局路径引力
                print("\n使用势场法进行局部避障规划...")
                
                # 修改：使用单步势场法规划，而不是生成完整路径
                # 这样更适合动态环境，只关注当前最佳移动方向
                next_joint_pos, local_cost = pf_planner.plan_next_step(current_joint_pos, goal_joint_pos, reference = False)

                print(f"计算下一步避障方向, 目标距离: {local_cost:.4f}")
                
                # 执行单步移动
                joint_indices = self.robot.arm_idx

                # 设置关节位置
                for i, idx in enumerate(joint_indices):
                    p.setJointMotorControl2(self.robot.id, idx, p.POSITION_CONTROL, next_joint_pos[i])

                # 执行若干仿真步骤
                for _ in range(steps):
                    self.sim.step()
                    time.sleep(delay)
            
                # 更新当前关节位置
                current_joint_pos = self.robot.get_joint_positions()

                # (f) 检查是否到达目标
                
                dist_to_goal = np.linalg.norm(np.array(current_joint_pos) - np.array(goal_joint_pos))
                goal_reached = dist_to_goal < 0.2  # 或与 planner.goal_threshold 一致

                if goal_reached:
                    print("\n目标位置到达!")
            
            print("\n基于 Potential Field 的轨迹执行完成。")

            ball_is_far = False
            while(not ball_is_far):
                rgb_static, depth_static, seg_static = self.sim.get_static_renders()
                detections = self.obstacle_tracker.detect_obstacles(rgb_static, depth_static, seg_static)
                tracked_positions = self.obstacle_tracker.update(detections)
                ball_is_far = self.obstacle_tracker.is_away()
                for _ in range(1):
                    self.sim.step()
                    time.sleep(1/240.)

            start_joint_pos = [0.9, 0.6099985502678034, 0.0026639666712291836, -0.47143263171620764, 0.0, 3.14*0.8, 2.9671]
            Path_start = SimpleTrajectoryPlanner.generate_joint_trajectory(current_joint_pos, start_joint_pos, steps=100)
            for joint_target in Path_start:
                    self.sim.robot.position_control(joint_target)
                    for _ in range(1):
                        self.sim.step()
                        time.sleep(1/240.)
            # 更新当前位置
            current_joint_pos = self.robot.get_joint_positions()

                
            print("\n轨迹执行完成, \n目标位置到达!")
            # Release object
            print("\nReleasing object...")
            self._release_object()
            
            print("Gripper opened, object placed at tray position")

        elif(method == "RRT*_PF_Plan"):
  

            start_pos = np.array([
                0.1,
                0.1,
                2.5
            ])
            start_orn = p.getQuaternionFromEuler([0, np.pi, 0])  # Vertically downward

            # if visualize:
            #     self._visualize_goal_position(start_pos)

            current_joint_pos = self.robot.get_joint_positions()

            start_joint_pos = self.ik_solver.solve(start_pos, start_orn, current_joint_pos, max_iters=50, tolerance=0.001)
            start_joint_pos[0] = 0.9

            Path_start = SimpleTrajectoryPlanner.generate_joint_trajectory(current_joint_pos, start_joint_pos, steps=100)
            for joint_target in Path_start:
                    self.sim.robot.position_control(joint_target)
                    for _ in range(10):
                        self.sim.step()
                        time.sleep(1/240.)
            current_joint_pos = self.robot.get_joint_positions()

            # ----------------- 1) 获取目标位置 -----------------
            goal_pos = np.array([0.45593274 ,0.55745936, 2.14598578])
            # 可视化目标位置（如需要）
            if visualize:
                self._visualize_goal_position(goal_pos)
            
            start_joint_pos = self.robot.get_joint_positions()
            goal_joint_pos = [0.9, 0.4099985502678034, 0.0026639666712291836, -0.67143263171620764, 0.0, 3.14*0.8, 2.9671]
            
            # ----------------- 2) 使用RRT*生成全局参考路径 -----------------
            print("\n使用RRT*生成全局参考路径...")
            
            # 初始化RRT*规划器
            rrt_planner = RRTStarPlanner(
                robot=self.robot,
                obstacle_tracker=self.obstacle_tracker,
                max_iterations=2000,
                step_size=0.1,
                goal_sample_rate=0.05,
                search_radius= 0.5,
                goal_threshold=0.05
            )
            
            # 使用静态相机获取障碍物位置
            rgb_static, depth_static, seg_static = self.sim.get_static_renders()
            detections = self.obstacle_tracker.detect_obstacles(rgb_static, depth_static, seg_static)
            tracked_positions = self.obstacle_tracker.update(detections)
            
            # 可视化障碍物边界框
            if visualize:
                self.obstacle_tracker.visualize_tracking_3d(tracked_positions)
                print(f"检测到 {len(tracked_positions)} 个障碍物")
            
            # 使用RRT*规划全局路径
            global_path, global_cost = rrt_planner.plan(start_joint_pos, goal_joint_pos)
            # 生成平滑轨迹
            global_path = rrt_planner.generate_smooth_trajectory(global_path, smoothing_steps=5) 
            if not global_path:
                print("无法生成全局RRT*路径，无法继续")
                return False
                
            print(f"成功生成全局参考路径! 路径代价: {global_cost:.4f}, 路径点数量: {len(global_path)}")
            
            # 可视化全局参考路径
            if visualize:
                self._visualize_path(rrt_planner, global_path)
                print("全局参考路径已可视化（绿色线条）")
            
            # ----------------- 3) 初始化势场规划器 -----------------
            print("\n初始化势场规划器用于实时避障...")
            
            pf_planner = PotentialFieldPlanner(
                robot_id=self.robot.id,
                joint_indices=self.robot.arm_idx,
                lower_limits=self.robot.lower_limits,
                upper_limits=self.robot.upper_limits,
                ee_link_index=self.robot.ee_idx,
                obstacle_tracker=self.obstacle_tracker,
                max_iterations=200,       # 每次规划迭代次数不需要太多
                step_size=0.01,           # 势场下降步长
                d0=0.25,                  # 排斥势生效距离
                K_att=5.0,                # 吸引势增益
                K_rep=100.0,                # 排斥势增益（加大以更好避障）
                goal_threshold=0.2,      # 到达目标的阈值
                collision_check_step=0.05,
                reference_path_weight=0.7  # 全局路径的引力权重
            )
            
            # 设置全局参考路径
            pf_planner.set_reference_path(global_path)
            
            # ----------------- 4) 执行动态避障主循环 -----------------
            print("\n开始执行基于RRT*-PF的动态避障...")
            
            # 调整执行速度参数
            steps = max(1, int( 10 * movement_speed_factor))
            delay = (1/240.0) * movement_speed_factor

            
            current_joint_pos = self.robot.get_joint_positions()
            goal_reached = False
            
            while not goal_reached:
                # (a) 获取当前环境并更新障碍物
                rgb_static, depth_static, seg_static = self.sim.get_static_renders()
                detections = self.obstacle_tracker.detect_obstacles(rgb_static, depth_static, seg_static)
                tracked_positions = self.obstacle_tracker.update(detections)
                
                # (b) 使用势场法重新规划，考虑全局路径引力
                print("\n使用势场法进行局部避障规划...")
                
                # 修改：使用单步势场法规划，而不是生成完整路径
                # 这样更适合动态环境，只关注当前最佳移动方向
                next_joint_pos, local_cost = pf_planner.plan_next_step(current_joint_pos, goal_joint_pos, reference = True)
                
                print(f"计算下一步避障方向, 目标距离: {local_cost:.4f}")
                
                # 执行单步移动
                joint_indices = self.robot.arm_idx
                
                # 设置关节位置
                for i, idx in enumerate(joint_indices):
                    p.setJointMotorControl2(self.robot.id, idx, p.POSITION_CONTROL, next_joint_pos[i])
                
                # 执行仿真步骤
                for _ in range(steps):
                    self.sim.step()
                    time.sleep(delay)
                
                # 更新当前关节位置
                current_joint_pos = self.robot.get_joint_positions()
                
                # (c) 检查是否到达目标
                dist_to_goal = np.linalg.norm(np.array(current_joint_pos) - np.array(goal_joint_pos))
                goal_reached = dist_to_goal < pf_planner.goal_threshold
                
                if goal_reached:
                    print("\n目标位置到达!")
            
            print("\nRRT*-PF动态避障轨迹执行完成")

            ball_is_far = False
            while(not ball_is_far):
                rgb_static, depth_static, seg_static = self.sim.get_static_renders()
                detections = self.obstacle_tracker.detect_obstacles(rgb_static, depth_static, seg_static)
                tracked_positions = self.obstacle_tracker.update(detections)
                ball_is_far = self.obstacle_tracker.is_away()
                for _ in range(1):
                    self.sim.step()
                    time.sleep(1/240.)

            start_joint_pos = [0.9, 0.6099985502678034, 0.0026639666712291836, -0.47143263171620764, 0.0, 3.14*0.8, 2.9671]
            Path_start = SimpleTrajectoryPlanner.generate_joint_trajectory(current_joint_pos, start_joint_pos, steps=100)
            for joint_target in Path_start:
                    self.sim.robot.position_control(joint_target)
                    for _ in range(1):
                        self.sim.step()
                        time.sleep(1/240.)
            # 更新当前位置
            current_joint_pos = self.robot.get_joint_positions()

                
            print("\n轨迹执行完成, \n目标位置到达!")
            # Release object
            print("\nReleasing object...")
            self._release_object()
            
            print("Gripper opened, object placed at tray position")

        return True
    
    def _visualize_goal_position(self, goal_pos):
        """Visualize target position"""        
        # Add coordinate axes at target position
        axis_length = 0.1  # 10cm long axes
        p.addUserDebugLine(
            goal_pos, 
            goal_pos + np.array([axis_length, 0, 0]), 
            [1, 0, 0], 3, 0  # X-axis - red
        )
        p.addUserDebugLine(
            goal_pos, 
            goal_pos + np.array([0, axis_length, 0]), 
            [0, 1, 0], 3, 0  # Y-axis - green
        )
        p.addUserDebugLine(
            goal_pos, 
            goal_pos + np.array([0, 0, axis_length]), 
            [0, 0, 1], 3, 0  # Z-axis - blue
        )
        
        # Add text label at target position
        p.addUserDebugText(
            f"Goal Position ({goal_pos[0]:.3f}, {goal_pos[1]:.3f}, {goal_pos[2]:.3f})",
            goal_pos + np.array([0, 0, 0.05]),  # Show text 5cm above the target position
            [1, 1, 1],  # White text
            1.0  # Text size
        )
    
    def _visualize_path(self, planner, path):
        """Visualize planned path"""
        # Clear previous visualization
        planner.clear_visualization()
        
        # Visualize path
        for i in range(len(path) - 1):
            start_ee, _ = planner._get_current_ee_pose(path[i])
            end_ee, _ = planner._get_current_ee_pose(path[i+1])
            
            p.addUserDebugLine(
                start_ee, end_ee, [0, 0, 1], 3, 0)
    
    def _execute_trajectory(self, joint_indices, trajectory, steps=5, delay=1/240.0):
        """Execute trajectory with adjustable speed
        
        Parameters:
        joint_indices: Joint indices to control
        trajectory: List of joint positions to execute
        steps: Number of simulation steps for each trajectory point (higher = slower movement)
        delay: Delay between steps (higher = slower movement)
        """
        for joint_pos in trajectory:
            # Set joint positions
            for i, idx in enumerate(joint_indices):
                p.setJointMotorControl2(self.robot.id, idx, p.POSITION_CONTROL, joint_pos[i])
            
            # Run multiple simulation steps for each trajectory point
            for _ in range(steps):
                self.sim.step()
                time.sleep(delay)
    
    def _release_object(self):
        """Release object"""
        # Open gripper
        open_gripper_width = 0.04  # Width to open the gripper
        p.setJointMotorControlArray(
            self.robot.id,
            jointIndices=self.robot.gripper_idx,
            controlMode=p.POSITION_CONTROL,
            targetPositions=[open_gripper_width, open_gripper_width]
        )
        
        # Wait for gripper to open
        for _ in range(int(1.0 * 240)):  # Wait for 1 second
            self.sim.step()
            time.sleep(1/240.)
