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
    
    def execute_planning(self, grasp_executor, planning_type='joint', visualize=True) -> bool:
        """
        Execute path planning
        
        Parameters:
        grasp_executor: Grasp executor object
        planning_type: Planning type ('joint' or 'cartesian')
        visualize: Whether to visualize the planning process
        
        Returns:
        success: Whether planning was successful
        """
        print(f"\nStep 4: {'Cartesian space' if planning_type == 'cartesian' else 'Joint space'} path planning...")
        
        # Get robot's current state (position after grasping) as starting point
        joint_indices = self.robot.arm_idx
        ee_link_index = self.robot.ee_idx
        
        # Get joint limits
        lower_limits = self.robot.lower_limits
        upper_limits = self.robot.upper_limits
        
        # Get current joint positions
        start_joint_pos = self.robot.get_joint_positions()
        
        # Get target tray position
        min_lim, max_lim = self.sim.goal._get_goal_lims()
        goal_pos = np.array([
            (min_lim[0] + max_lim[0])/2 - 0.1,
            (min_lim[1] + max_lim[1])/2 - 0.1,
            max_lim[2] + 0.2
        ])
        print(f"Tray target position: {goal_pos}")
        
        # Target position and orientation (above the tray, at a reasonable distance)
        tray_approach_pos = goal_pos.copy()  # Position above the tray
        tray_orn = p.getQuaternionFromEuler([0, np.pi, 0])  # Vertically downward
        
        # Visualize tray target position in PyBullet
        if visualize:
            self._visualize_goal_position(goal_pos)
        
        # Use static camera to get obstacle positions
        rgb_static, depth_static, seg_static = self.sim.get_static_renders()
        detections = self.obstacle_tracker.detect_obstacles(rgb_static, depth_static, seg_static)
        tracked_positions = self.obstacle_tracker.update(detections)
        
        # Visualize obstacle bounding boxes (if needed)
        if visualize:
            bounding_box_ids = self.obstacle_tracker.visualize_tracking_3d(tracked_positions)
            print(f"Detected {len(tracked_positions)} obstacles")
        
        # Choose and use appropriate planner based on planning type
        if planning_type == 'cartesian':
            # Use Cartesian space planning
            path, cost = self._execute_cartesian_planning(
                start_joint_pos, tray_approach_pos, tray_orn, visualize
            )
        else:
            # Use joint space planning
            path, cost = self._execute_joint_planning(
                start_joint_pos, tray_approach_pos, tray_orn, visualize
            )
        
        if not path:
            print("No path found")
            return False
        
        print(f"Path found! Cost: {cost:.4f}, Number of path points: {len(path)}")
        
        # Get the planner used
        planner = self._get_planner(planning_type)
        
        # Visualize trajectory
        if visualize and planner:
            self._visualize_path(planner, path)
        
        # Generate smooth trajectory
        print("\nGenerating smooth trajectory...")
        smooth_path = planner.generate_smooth_trajectory(path, smoothing_steps=20)
        
        # Execute trajectory
        print("\nExecuting trajectory...")
        self._execute_trajectory(joint_indices, smooth_path)
        
        print("\nPath execution completed")
        
        # Release object
        print("\nReleasing object...")
        self._release_object()
        
        print("Gripper opened, object placed at tray position")
        
        return True
    
    def _execute_cartesian_planning(self, start_joint_pos, goal_pos, goal_orn, visualize):
        """Execute Cartesian space planning"""
        # Create Cartesian space planner
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
        
        # Plan in Cartesian space
        print(f"\nPlanning using Cartesian space RRT*...")
        return planner.plan(start_joint_pos, goal_pos, goal_orn)
    
    def _execute_joint_planning(self, start_joint_pos, goal_pos, goal_orn, visualize):
        """Execute joint space planning"""
        # Create joint space planner
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
        
        # Try to convert target Cartesian position to joint space
        try:
            goal_joint_pos = self.ik_solver.solve(
                goal_pos, goal_orn, start_joint_pos, max_iters=50, tolerance=0.001
            )
            print(f"Target position IK solution: {goal_joint_pos}")
        except Exception as e:
            print(f"Unable to find IK solution for target position: {e}")
            return None, 0
        
        # Plan in joint space
        print(f"\nPlanning using joint space RRT*...")
        return planner.plan(start_joint_pos, goal_joint_pos)
    
    def _get_planner(self, planning_type):
        """Get the corresponding planner instance based on planning type"""
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
        """Visualize target position"""
        visual_id = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=0.03,  # 3cm radius sphere
            rgbaColor=[0, 0, 1, 0.7]  # Blue semi-transparent
        )
        goal_marker_id = p.createMultiBody(
            baseMass=0,  # Mass of 0 indicates static object
            baseVisualShapeIndex=visual_id,
            basePosition=goal_pos.tolist()
        )
        
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
    
    def _execute_trajectory(self, joint_indices, trajectory):
        """Execute trajectory"""
        for joint_pos in trajectory:
            # Set joint positions
            for i, idx in enumerate(joint_indices):
                p.setJointMotorControl2(self.robot.id, idx, p.POSITION_CONTROL, joint_pos[i])
            
            # Update simulation
            self.sim.step()
            time.sleep(0.01)
    
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
