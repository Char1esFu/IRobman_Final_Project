import numpy as np
import pybullet as p

from typing import List, Tuple

from src.robot import Robot
from src.obstacle_tracker.obstacle_tracker import ObstacleTracker



class PotentialFieldPlanner:
    """
    Potential Field Planner for robotic arm.
    
    Plans in joint space while performing collision detection in Cartesian space.
    Can be used for dynamic obstacle avoidance in combination with global RRT* path.
    
    Args:
        robot: Instance of Robot class
        obstacle_tracker: Instance of ObstacleTracker to get obstacle positions
        max_iterations: Maximum number of iterations for potential field descent
        step_size: Step size for gradient descent
        d0: Influence distance of obstacles
        K_att: Attraction gain
        K_rep: Repulsion gain
        goal_threshold: Distance threshold to consider goal reached (joint space)
        collision_check_step: Step size for collision checking along the path
        reference_path_weight: Weight for reference path attraction (for RRT* path following)
    """
    def __init__(
        self,
        robot: Robot,
        obstacle_tracker: ObstacleTracker,
        max_iterations: int = 300,
        step_size: float = 0.01,
        d0: float = 0.2,
        K_att: float = 1.0,
        K_rep: float = 1.0,
        goal_threshold: float = 0.05,
        collision_check_step: float = 0.05,
        reference_path_weight: float = 0.5
    ):
        self.robot = robot
        self.obstacle_tracker = obstacle_tracker
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.d0 = d0
        self.K_att = K_att
        self.K_rep = K_rep
        self.goal_threshold = goal_threshold
        self.collision_check_step = collision_check_step
        self.reference_path_weight = reference_path_weight
        
        self.dimension = len(robot.arm_idx)
        
        # Visualization lines
        self.debug_lines = []
        
        # Store possible global reference trajectory
        self.reference_path = None
    
    def _get_current_ee_pose(self, joint_positions: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Get the end effector pose at specified joint positions
        
        Args:
            joint_positions: Joint positions
            
        Returns:
            Tuple of end effector position and orientation
        """
        # Save current state
        current_states = []
        for i in self.robot.arm_idx:
            current_states.append(p.getJointState(self.robot.id, i)[0])
            
        # Set joint positions
        for i, idx in enumerate(self.robot.arm_idx):
            p.resetJointState(self.robot.id, idx, joint_positions[i])
            
        # Get end effector pose
        ee_state = p.getLinkState(self.robot.id, self.robot.ee_idx)
        ee_pos = np.array(ee_state[0])
        ee_orn = np.array(ee_state[1])
        
        # Restore original state
        for i, idx in enumerate(self.robot.arm_idx):
            p.resetJointState(self.robot.id, idx, current_states[i])
            
        return ee_pos, ee_orn
    
    def _is_collision_free(self, joint_pos: List[float]) -> bool:
        """Check if joint positions are collision free
        
        Args:
            joint_pos: Joint positions to check
            
        Returns:
            True if collision free, False otherwise
        """
        # Get end effector position and orientation
        ee_pos, _ = self._get_current_ee_pose(joint_pos)
        
        # Get robot links positions to check
        links_to_check = self.robot.arm_idx + [self.robot.ee_idx]
        
        # Save current state
        current_states = []
        for i in self.robot.arm_idx:
            current_states.append(p.getJointState(self.robot.id, i)[0])
            
        # Set joint positions
        for i, idx in enumerate(self.robot.arm_idx):
            p.resetJointState(self.robot.id, idx, joint_pos[i])
        
        # Check obstacle collision
        collision = False
        
        # Get obstacle states
        obstacle_states = self.obstacle_tracker.get_all_obstacle_states()
        if obstacle_states is None or len(obstacle_states) == 0:
            # Restore original state
            for i, idx in enumerate(self.robot.arm_idx):
                p.resetJointState(self.robot.id, idx, current_states[i])
            return True  # No obstacles, no collision
        
        # Check collision between each link and each obstacle
        for link_idx in links_to_check:
            link_state = p.getLinkState(self.robot.id, link_idx)
            link_pos = np.array(link_state[0])
            
            for obstacle in obstacle_states:
                if obstacle is None:
                    continue
                
                # Simple sphere collision check
                obstacle_pos = obstacle['position']
                obstacle_radius = obstacle['radius']
                
                # Distance between link and obstacle center
                dist = np.linalg.norm(link_pos - obstacle_pos)
                
                # Approximate robot links as points (simplified)
                # Add small safety margin (0.05m)
                if dist < obstacle_radius + 0.05:
                    collision = True
                    break
            
            if collision:
                break
                
        # Restore original state
        for i, idx in enumerate(self.robot.arm_idx):
            p.resetJointState(self.robot.id, idx, current_states[i])
            
        return not collision
    
    def _distance(self, q1: List[float], q2: List[float]) -> float:
        """Calculate distance between two joint configurations
        
        Args:
            q1: First joint configuration
            q2: Second joint configuration
            
        Returns:
            Euclidean distance in joint space
        """
        return np.linalg.norm(np.array(q1) - np.array(q2))
    

    
    def _attractive_gradient(self, q: np.ndarray, q_goal: np.ndarray) -> np.ndarray:
        """Calculate attractive potential gradient
        
        Args:
            q: Current joint configuration
            q_goal: Goal joint configuration
            
        Returns:
            Attractive potential gradient (pointing to goal)
        """
        return self.K_att * (q - q_goal)
    
    def _repulsive_potential(self, q: np.ndarray) -> float:
        """Calculate repulsive potential
        
        Args:
            q: Current joint configuration
            
        Returns:
            Repulsive potential value
        """
        # Save current state
        current_states = []
        for i in self.robot.arm_idx:
            current_states.append(p.getJointState(self.robot.id, i)[0])
            
        # Set joint positions
        for i, idx in enumerate(self.robot.arm_idx):
            p.resetJointState(self.robot.id, idx, q[i])
        
        # Links to check
        links_to_check = self.robot.arm_idx + [self.robot.ee_idx]
        
        # Get obstacle states
        obstacle_states = self.obstacle_tracker.get_all_obstacle_states()
        
        potential = 0.0
        
        if obstacle_states:
            for link_idx in links_to_check:
                link_state = p.getLinkState(self.robot.id, link_idx)
                link_pos = np.array(link_state[0])
                
                for obstacle in obstacle_states:
                    if obstacle is None:
                        continue
                    
                    obstacle_pos = obstacle['position']
                    obstacle_radius = obstacle['radius']
                    
                    # Distance between link and obstacle center
                    dist = np.linalg.norm(link_pos - obstacle_pos) - obstacle_radius
                    
                    # If within influence range, calculate repulsive potential
                    if dist < self.d0:
                        if dist < 0.01:  # Prevent division by very small number
                            dist = 0.01
                        
                        # Repulsive potential formula: 0.5 * K_rep * (1/dist - 1/d0)^2
                        potential += 0.5 * self.K_rep * ((1.0 / dist) - (1.0 / self.d0))**2
        
        # Restore original state
        for i, idx in enumerate(self.robot.arm_idx):
            p.resetJointState(self.robot.id, idx, current_states[i])
            
        return potential
    
    def _repulsive_gradient(self, q: np.ndarray) -> np.ndarray:
        """Calculate repulsive potential gradient
        
        Args:
            q: Current joint configuration
            
        Returns:
            Repulsive potential gradient (pointing away from obstacles)
        """
        # Numerical differentiation to calculate gradient
        grad = np.zeros(self.dimension)
        epsilon = 1e-3  # Small perturbation
        
        for i in range(self.dimension):
            q_plus = q.copy()
            q_plus[i] += epsilon
            
            q_minus = q.copy()
            q_minus[i] -= epsilon
            
            # Central difference
            grad[i] = (self._repulsive_potential(q_plus) - self._repulsive_potential(q_minus)) / (2 * epsilon)
        
        return grad
    
    def _reference_path_gradient(self, q: np.ndarray) -> np.ndarray:
        """Calculate reference path gradient (guide robot to follow RRT* global path)
        
        Args:
            q: Current joint configuration
            
        Returns:
            Reference path gradient
        """
        if self.reference_path is None or len(self.reference_path) < 2:
            return np.zeros(self.dimension)
        
        # Find the nearest point on the reference path
        distances = [self._distance(q, np.array(p)) for p in self.reference_path]
        min_idx = np.argmin(distances)
        
        # If already the last point, point to the last point
        if min_idx == len(self.reference_path) - 1:
            closest_point = np.array(self.reference_path[min_idx])
            return self.reference_path_weight * self.K_att * (q - closest_point)
        
        # Otherwise, point to the next path point
        next_point = np.array(self.reference_path[min_idx + 1])
        return self.reference_path_weight * self.K_att * (q - next_point)
    
    def _total_gradient(self, q: np.ndarray, q_goal: np.ndarray ,reference) -> np.ndarray:
        """Calculate total gradient (attractive + repulsive + reference path)
        
        Args:
            q: Current joint configuration
            q_goal: Goal joint configuration
            
        Returns:
            Total gradient
        """
        # Attractive gradient (pointing to goal)
        att_grad = -self._attractive_gradient(q, q_goal)
        
        # Repulsive gradient (pointing away from obstacles)
        rep_grad = -self._repulsive_gradient(q)
        
        if reference :
            # Reference path gradient (guide following RRT* path)
            ref_grad = -self._reference_path_gradient(q)
            # Total gradient = attractive + repulsive + reference path
            total_grad = att_grad + rep_grad + ref_grad
        else:
            total_grad = att_grad + rep_grad
        
        # Normalize gradient
        norm = np.linalg.norm(total_grad)
        if norm > 1e-6:  # Prevent division by zero
            total_grad = total_grad / norm
        
        return total_grad
    
    
    def plan_next_step(self, current_config: List[float], goal_config: List[float], reference: bool) -> Tuple[List[float], float]:
        """Calculate the next best step from the current position
        
        This method is more suitable for dynamic environments, focusing only on the current local optimal direction
        
        Args:
            current_config: Current joint configuration
            goal_config: Goal joint configuration
            reference: Whether there is a reference trajectory
            
        Returns:
            Tuple (next joint configuration, distance to goal)
        """
        q_current = np.array(current_config)
        q_goal = np.array(goal_config)
        
        # Calculate gradient direction
        gradient = self._total_gradient(q_current, q_goal,reference)
        
        # Update position based on gradient
        q_new = q_current + self.step_size * gradient
        
        # Force within joint limits
        for j in range(self.dimension):
            q_new[j] = max(self.robot.lower_limits[j], min(self.robot.upper_limits[j], q_new[j]))
        
        # Check if collision free, if not try using only repulsive force
        if not self._is_collision_free(q_new):
            rep_grad = -self._repulsive_gradient(q_current)
            norm_rep = np.linalg.norm(rep_grad)
            if norm_rep > 1e-6:
                rep_grad = rep_grad / norm_rep
                q_new = q_current + self.step_size * rep_grad
                
                # Force within joint limits
                for j in range(self.dimension):
                    q_new[j] = max(self.robot.lower_limits[j], min(self.robot.upper_limits[j], q_new[j]))
                
                # If still collision, return current position
                if not self._is_collision_free(q_new):
                    q_new = q_current
        
        # Calculate distance to goal
        ee_pos, _ = self._get_current_ee_pose(q_new.tolist())
        goal_ee_pos, _ = self._get_current_ee_pose(goal_config)
        cost = np.linalg.norm(ee_pos - goal_ee_pos)
        
        # Visualize this step
        start_ee, _ = self._get_current_ee_pose(current_config)
        end_ee, _ = self._get_current_ee_pose(q_new.tolist())
        
        debug_id = p.addUserDebugLine(
            start_ee, end_ee, [1, 0, 1], 2, 0  # Purple line represents potential field path
        )
        
        self.debug_lines.append(debug_id)
        
        return q_new.tolist(), cost
    
    def clear_visualization(self) -> None:
        """Clear path visualization"""
        for debug_id in self.debug_lines:
            p.removeUserDebugItem(debug_id)
        self.debug_lines = []
    
    def set_reference_path(self, reference_path: List[List[float]]) -> None:
        """Set reference path (for RRT* and potential field method combination)
        
        Args:
            reference_path: Reference path generated by RRT*
        """
        self.reference_path = reference_path
        print(f"Set reference path, containing {len(reference_path)} points")
    
    