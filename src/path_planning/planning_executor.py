import numpy as np
import pybullet as p
import time
from typing import Any, Dict

from src.path_planning.rrt_star import RRTStarPlanner
from src.path_planning.rrt_star_cartesian import RRTStarCartesianPlanner
from src.path_planning.potential_field import PotentialFieldPlanner  
from src.obstacle_tracker.obstacle_tracker import ObstacleTracker
from src.ik_solver.ik_solver import DifferentialIKSolver
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
        print(f"Dynamic replanning: {'Enabled' if enable_replan else 'Disabled'}")
        
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
            start_joint_pos[0] = 0.7

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

            start_joint_pos = [0.7, 0.4099985502678034, 0.0026639666712291836, -0.67143263171620764, 0.0, 3.14*0.8, 2.9671]
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

   
            start_joint_pos = [0.7, 0.6099985502678034, 0.0026639666712291836, -0.47143263171620764, 0.0, 3.14*0.8, 2.9671]
            Path_start = SimpleTrajectoryPlanner.generate_joint_trajectory(current_joint_pos, start_joint_pos, steps=100)
            for joint_target in Path_start:
                    self.sim.robot.position_control(joint_target)
                    for _ in range(1):
                        self.sim.step()
                        time.sleep(1/240.)
            # update current position
            current_joint_pos = self.robot.get_joint_positions()

                
            print("\nTrajectory execution completed, \nTarget position reached!")
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
            
            # calculate target joint position (only needed in joint space planning)
            if planning_type == 'joint':
                goal_joint_pos = self.ik_solver.solve(
                    goal_pos, goal_orn, self.robot.get_joint_positions(), max_iters=50, tolerance=0.001
                )
            
            # if dynamic replanning is enabled, we will track the current position and target position
            if enable_replan:
                # initialize planner (will be reused in the loop)
                if planning_type == 'cartesian':
                    planner = RRTStarCartesianPlanner(
                        robot_id=self.robot.id,
                        joint_indices=self.robot.arm_idx,
                        lower_limits=self.robot.lower_limits,
                        upper_limits=self.robot.upper_limits,
                        ee_link_index=self.robot.ee_idx,
                        obstacle_tracker=self.obstacle_tracker,
                        max_iterations=500,  # reduce iterations for replanning to speed up
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
                
                # main loop: execute path, monitor obstacles and replan
                print("\nStarting trajectory execution with dynamic replanning...")
                
                # adjust execution speed parameters
                steps = max(1, int(10 * movement_speed_factor))
                delay = (1/240.0) * movement_speed_factor
                
                current_joint_pos = self.robot.get_joint_positions()
                goal_reached = False
                
                while not goal_reached:
                    # update obstacle positions
                    rgb_static, depth_static, seg_static = self.sim.get_static_renders()
                    detections = self.obstacle_tracker.detect_obstacles(rgb_static, depth_static, seg_static)
                    tracked_positions = self.obstacle_tracker.update(detections)
                    
                    # # visualize obstacle bounding boxes
                    # if visualize:
                    #     self.obstacle_tracker.visualize_tracking_3d(tracked_positions)
                    #     print(f"Detected {len(tracked_positions)} obstacles")
                    
                    # replan from current position to target
                    print("\nReplanning path...")
                    if planning_type == 'cartesian':
                        path, cost = planner.plan(current_joint_pos, goal_pos, goal_orn)
                    else:  # joint space planning
                        path, cost = planner.plan(current_joint_pos, goal_joint_pos)
                    
                    if not path:
                        print("No valid path found, trying again...")
                        time.sleep(0.5)  # wait a moment and try again
                        continue
                    
                    print(f"Path found! Cost: {cost:.4f}, Number of path points: {len(path)}")
                    
                    # visualize trajectory
                    if visualize and planner:
                        self._visualize_path(planner, path)
                    
                    # generate smooth trajectory
                    smooth_path = planner.generate_smooth_trajectory(path, smoothing_steps=5)
                    
                    # execute only a part of the trajectory, then replan
                    subpath = smooth_path[:min(replan_steps, len(smooth_path))]
                    
                    # execute a part of the trajectory, then replan
                    for joint_pos in subpath:
                        # set joint position
                        for i, idx in enumerate(joint_indices):
                            p.setJointMotorControl2(self.robot.id, idx, p.POSITION_CONTROL, joint_pos[i])
                        
                        # execute multiple simulation steps
                        for _ in range(steps):
                            self.sim.step()
                            time.sleep(delay)
                        
                        # update current position
                        current_joint_pos = self.robot.get_joint_positions()
                    
                    # check if target is reached
                    if planning_type == 'joint':
                        dist_to_goal = np.linalg.norm(np.array(current_joint_pos) - np.array(goal_joint_pos))
                        goal_reached = dist_to_goal < planner.goal_threshold
                    else:  # cartesian space
                        current_ee_pos, _ = self.robot.get_ee_pose()
                        dist_to_goal = np.linalg.norm(np.array(current_ee_pos) - np.array(goal_pos))
                        goal_reached = dist_to_goal < 0.03  # 厘米级精度
                    
                    if goal_reached:
                        print("\nTarget position reached!")
                
                print("\nTrajectory execution completed")
                
            else:
                # without dynamic replanning
                # Use static camera to get obstacle positions
                rgb_static, depth_static, seg_static = self.sim.get_static_renders()
                detections = self.obstacle_tracker.detect_obstacles(rgb_static, depth_static, seg_static)
                tracked_positions = self.obstacle_tracker.update(detections)
                
                # Visualize obstacle bounding boxes (if needed)
                if visualize:
                    bounding_box_ids = self.obstacle_tracker.visualize_tracking_3d(tracked_positions)
                    print(f"Detected {len(tracked_positions)} obstacles")
                
                # get current joint position
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
                # adjust steps and delay based on speed factor
                steps = int(10 * movement_speed_factor)
                delay = (1/240.0) * movement_speed_factor
                self._execute_trajectory(joint_indices, smooth_path, steps=steps, delay=delay)
                
                print("\nPath execution completed")

            # update current position
            current_joint_pos = self.robot.get_joint_positions()
            
            # check if target is reached
            if planning_type == 'joint':
                dist_to_goal = np.linalg.norm(np.array(current_joint_pos) - np.array(goal_joint_pos))
                goal_reached = dist_to_goal < planner.goal_threshold
            else:  # cartesian space
                current_ee_pos, _ = self.robot.get_ee_pose()
                dist_to_goal = np.linalg.norm(np.array(current_ee_pos) - np.array(goal_pos))
                goal_reached = dist_to_goal < 0.03  # 厘米级精度
            
            if goal_reached:
                print("\nTarget position reached!")
                
            print("\nTrajectory execution completed")
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
            # ----------------- 1) Get target information -----------------
            goal_pos = np.array([0.54939217, 0.46558996, 2.14602761])
            # visualize target position (if needed)
            if visualize:
                self._visualize_goal_position(goal_pos)
            
        
            goal_joint_pos = [0.7, 0.4099985502678034, 0.0026639666712291836, -0.67143263171620764, 0.0, 3.14*0.8, 2.9671]
            
            # ----------------- 2) Initialize potential field planner -----------------
            print("\nInitialize potential field planner for real-time obstacle avoidance...")
            
            pf_planner = PotentialFieldPlanner(
                robot=self.robot,
                obstacle_tracker=self.obstacle_tracker,
                max_iterations=200,
                step_size=0.01,           # potential field descent step
                d0=0.5,                  # repulsive force effective distance
                K_att=1.0,                # attractive force gain
                K_rep=100.0,                # repulsive force gain
                goal_threshold=0.05,      # target threshold
                collision_check_step=0.2,
                reference_path_weight=0.7  # global path attractive force weight
            )

            # ----------------- 3) Dynamic replanning main loop -----------------
            print("\nStart executing dynamic replanning based on Potential Field...")
            steps = max(1, int(10 * movement_speed_factor))
            delay = (1 / 240.0) * movement_speed_factor
            
            current_joint_pos = self.robot.get_joint_positions()
            goal_reached = False

            while not goal_reached:
                # Get current environment image and update obstacles
                rgb_static, depth_static, seg_static = self.sim.get_static_renders()
                detections = self.obstacle_tracker.detect_obstacles(rgb_static, depth_static, seg_static)
                tracked_positions = self.obstacle_tracker.update(detections)

                # Use potential field method to replan, considering global path attractive force
                print("\nUse potential field method to local obstacle avoidance planning...")
                
                next_joint_pos, local_cost = pf_planner.plan_next_step(current_joint_pos, goal_joint_pos, reference = False)

                print(f"Calculate next obstacle avoidance direction, target distance: {local_cost:.4f}")
                
                joint_indices = self.robot.arm_idx

                # Set joint position
                for i, idx in enumerate(joint_indices):
                    p.setJointMotorControl2(self.robot.id, idx, p.POSITION_CONTROL, next_joint_pos[i])

                # Execute several simulation steps
                for _ in range(steps):
                    self.sim.step()
                    time.sleep(delay)
            
                # Update current joint position
                current_joint_pos = self.robot.get_joint_positions()

                # Check if target is reached
                
                dist_to_goal = np.linalg.norm(np.array(current_joint_pos) - np.array(goal_joint_pos))
                goal_reached = dist_to_goal < 0.2  # or consistent with planner.goal_threshold

                if goal_reached:
                    print("\nTarget position reached!")
            
            print("\nPotential field trajectory execution completed.")

            ball_is_far = False
            while(not ball_is_far):
                rgb_static, depth_static, seg_static = self.sim.get_static_renders()
                detections = self.obstacle_tracker.detect_obstacles(rgb_static, depth_static, seg_static)
                tracked_positions = self.obstacle_tracker.update(detections)
                ball_is_far = self.obstacle_tracker.is_away()
                for _ in range(1):
                    self.sim.step()
                    time.sleep(1/240.)

            start_joint_pos = [0.7, 0.6099985502678034, 0.0026639666712291836, -0.47143263171620764, 0.0, 3.14*0.8, 2.9671]
            Path_start = SimpleTrajectoryPlanner.generate_joint_trajectory(current_joint_pos, start_joint_pos, steps=100)
            for joint_target in Path_start:
                    self.sim.robot.position_control(joint_target)
                    for _ in range(1):
                        self.sim.step()
                        time.sleep(1/240.)
            # Update current position
            current_joint_pos = self.robot.get_joint_positions()

                
            print("\nTrajectory execution completed, \nTarget position reached!")
            # Release object
            print("\nReleasing object...")
            self._release_object()
            
            print("Gripper opened, object placed at tray position")

        elif(method == "RRT*_PF_Plan"):

            current_joint_pos = self.robot.get_joint_positions()
            start_joint_pos = [-1.8, -0.1834053988761717, 0.30703679700804354, -0.4583896277866867, 0.2661630896757948, 0.49144743727571705, -2.0497514664232424]
            
            Path_start = SimpleTrajectoryPlanner.generate_joint_trajectory(current_joint_pos, start_joint_pos, steps=100)
            for joint_target in Path_start:
                    self.sim.robot.position_control(joint_target)
                    for _ in range(10):
                        self.sim.step()
                        time.sleep(1/240.)

            current_joint_pos = self.robot.get_joint_positions()

            start_joint_pos = [0.9, -0.1834053988761717, 0.30703679700804354, -0.4583896277866867, 0.2661630896757948, 0.49144743727571705, -2.0497514664232424]
            
            Path_start = SimpleTrajectoryPlanner.generate_joint_trajectory(current_joint_pos, start_joint_pos, steps=100)
            for joint_target in Path_start:
                    self.sim.robot.position_control(joint_target)
                    for _ in range(10):
                        self.sim.step()
                        time.sleep(1/240.)
            current_joint_pos = self.robot.get_joint_positions()

            # ----------------- 1) 获取目标位置 -----------------
            goal_pos = np.array([0.54939217, 0.46558996, 2.14602761])
            # Visualize target position (if needed)
            if visualize:
                self._visualize_goal_position(goal_pos)
            
            start_joint_pos = self.robot.get_joint_positions()
            goal_joint_pos = [0.7, 0.4099985502678034, 0.0026639666712291836, -0.67143263171620764, 0.0, 3.14*0.8, 2.9671]
            
            # ----------------- 2) 使用RRT*生成全局参考路径 -----------------
            print("\n使用RRT*生成全局参考路径...")
            
            # Initialize RRT* planner
            rrt_planner = RRTStarPlanner(
                robot=self.robot,
                obstacle_tracker=self.obstacle_tracker,
                max_iterations=2000,
                step_size=0.1,
                goal_sample_rate=0.05,
                search_radius= 0.5,
                goal_threshold=0.05
            )
            
            # Get obstacle positions using static camera
            rgb_static, depth_static, seg_static = self.sim.get_static_renders()
            detections = self.obstacle_tracker.detect_obstacles(rgb_static, depth_static, seg_static)
            tracked_positions = self.obstacle_tracker.update(detections)
            
            # Visualize obstacle bounding boxes
            if visualize:
                self.obstacle_tracker.visualize_tracking_3d(tracked_positions)
                print(f"Detected {len(tracked_positions)} obstacles")
            
            # Use RRT* to plan global path
            global_path, global_cost = rrt_planner.plan(start_joint_pos, goal_joint_pos)
            # Generate smooth trajectory
            global_path = rrt_planner.generate_smooth_trajectory(global_path, smoothing_steps=5) 
            if not global_path:
                print("Cannot generate global RRT* path, cannot continue")
                return False
                
            print(f"Successfully generated global reference path! Path cost: {global_cost:.4f}, Path point number: {len(global_path)}")
            
            # Visualize global reference path
            if visualize:
                self._visualize_path(rrt_planner, global_path)
                print("Global reference path visualized (green lines)")
            
            # ----------------- 3) Initialize potential field planner -----------------
            print("\nInitialize potential field planner for real-time obstacle avoidance...")
            
            pf_planner = PotentialFieldPlanner(
                robot=self.robot,
                obstacle_tracker=self.obstacle_tracker,
                max_iterations=200,
                step_size=0.01,
                d0=0.25,
                K_att=5.0,
                K_rep=100.0,
                goal_threshold=0.2,
                collision_check_step=0.05,
                reference_path_weight=0.2
            )
            
            # Set global reference path
            pf_planner.set_reference_path(global_path)
            
            # ----------------- 4) Execute dynamic obstacle avoidance main loop -----------------
            print("\nStart executing dynamic obstacle avoidance based on RRT*-PF...")
            
            # Adjust execution speed parameters
            steps = max(1, int( 10 * movement_speed_factor))
            delay = (1/240.0) * movement_speed_factor

            
            current_joint_pos = self.robot.get_joint_positions()
            goal_reached = False
            
            while not goal_reached:
                # Get current environment and update obstacles
                rgb_static, depth_static, seg_static = self.sim.get_static_renders()
                detections = self.obstacle_tracker.detect_obstacles(rgb_static, depth_static, seg_static)
                tracked_positions = self.obstacle_tracker.update(detections)
                
                # Use potential field method to replan, considering global path attractive force
                print("\nUse potential field method to local obstacle avoidance planning...")
                
                next_joint_pos, local_cost = pf_planner.plan_next_step(current_joint_pos, goal_joint_pos, reference = True)
                
                print(f"Calculate next obstacle avoidance direction, target distance: {local_cost:.4f}")
                
                joint_indices = self.robot.arm_idx
                
                # Set joint position
                for i, idx in enumerate(joint_indices):
                    p.setJointMotorControl2(self.robot.id, idx, p.POSITION_CONTROL, next_joint_pos[i])
                
                # Execute simulation steps
                for _ in range(steps):
                    self.sim.step()
                    time.sleep(delay)
                
                # Update current joint position
                current_joint_pos = self.robot.get_joint_positions()
                
                # Check if target is reached
                dist_to_goal = np.linalg.norm(np.array(current_joint_pos) - np.array(goal_joint_pos))
                goal_reached = dist_to_goal < pf_planner.goal_threshold
                
                if goal_reached:
                    print("\nTarget position reached!")
            
            print("\nRRT*-PF dynamic obstacle avoidance trajectory execution completed")

            ball_is_far = False
            while(not ball_is_far):
                rgb_static, depth_static, seg_static = self.sim.get_static_renders()
                detections = self.obstacle_tracker.detect_obstacles(rgb_static, depth_static, seg_static)
                tracked_positions = self.obstacle_tracker.update(detections)
                ball_is_far = self.obstacle_tracker.is_away()
                for _ in range(1):
                    self.sim.step()
                    time.sleep(1/240.)

            start_joint_pos = [0.7, 0.6099985502678034, 0.0026639666712291836, -0.47143263171620764, 0.0, 3.14*0.8, 2.9671]
            Path_start = SimpleTrajectoryPlanner.generate_joint_trajectory(current_joint_pos, start_joint_pos, steps=100)
            for joint_target in Path_start:
                    self.sim.robot.position_control(joint_target)
                    for _ in range(1):
                        self.sim.step()
                        time.sleep(1/240.)
            # Update current position
            current_joint_pos = self.robot.get_joint_positions()

                
            print("\nTrajectory execution completed, \nTarget position reached!")
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
