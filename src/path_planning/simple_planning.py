import numpy as np
import pybullet as p
from typing import List, Tuple, Optional

class SimpleTrajectoryPlanner:
    """
    Trajectory planner for generating robot movement trajectories
    """
    
    @staticmethod
    def generate_joint_trajectory(start_joints: List[float], end_joints: List[float], steps: int = 100) -> List[List[float]]:
        """
        Generate a smooth trajectory from start to end joint positions
        
        Parameters:
        start_joints: Starting joint positions
        end_joints: Ending joint positions
        steps: Number of interpolation steps
        
        Returns:
        trajectory: List of joint positions
        """
        trajectory = []
        for step in range(steps + 1):
            t = step / steps  # Normalized step
            # Linear interpolation
            point = [start + t * (end - start) for start, end in zip(start_joints, end_joints)]
            trajectory.append(point)
        return trajectory
    
    @staticmethod
    def generate_cartesian_trajectory(robot_id: int, 
                                    arm_idx: List[int], 
                                    ee_idx: int, 
                                    start_joints: List[float], 
                                    target_pos: np.ndarray, 
                                    target_orn: List[float], 
                                    steps: int = 100) -> List[List[float]]:
        """
        Generate a linear trajectory in Cartesian space
        
        Parameters:
        robot_id: Robot ID
        arm_idx: List of robot arm joint indices
        ee_idx: End effector index
        start_joints: Starting joint positions
        target_pos: Target position
        target_orn: Target orientation
        steps: Number of interpolation steps
        
        Returns:
        trajectory: List of joint positions
        """
        # Set starting position
        for i, joint_idx in enumerate(arm_idx):
            p.resetJointState(robot_id, joint_idx, start_joints[i])
        
        # Get current end effector pose
        ee_state = p.getLinkState(robot_id, ee_idx)
        start_pos = np.array(ee_state[0])
        
        # Generate linear trajectory
        trajectory = []
        
        # Initialize IK solver
        from src.ik_solver import DifferentialIKSolver
        ik_solver = DifferentialIKSolver(robot_id, ee_idx, damping=0.05)
        
        for step in range(steps + 1):
            t = step / steps  # Normalized step
            
            # Linear interpolation
            pos = start_pos + t * (target_pos - start_pos)
            
            # Solve IK for current Cartesian position
            current_joints = ik_solver.solve(pos, target_orn, start_joints, max_iters=50, tolerance=0.001)
            
            # Add solution to trajectory
            trajectory.append(current_joints)
            
            # Reset to starting position
            for i, joint_idx in enumerate(arm_idx):
                p.resetJointState(robot_id, joint_idx, start_joints[i])
        
        return trajectory
