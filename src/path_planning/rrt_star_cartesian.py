import numpy as np
import pybullet as p
import random
import time
from scipy.spatial import KDTree
from typing import List, Tuple, Dict, Optional, Any, Callable
from src.ik_solver import DifferentialIKSolver

class RRTStarCartesianPlanner:
    """RRT* path planning algorithm in Cartesian space for robotic arm.
    
    Plans in Cartesian space while performing collision detection and IK conversion.
    
    Args:
        robot_id: PyBullet robot ID
        joint_indices: List of joint indices to control
        lower_limits: Lower joint limits
        upper_limits: Upper joint limits
        ee_link_index: End effector link index
        obstacle_tracker: Instance of ObstacleTracker to get obstacle positions
        max_iterations: Maximum number of RRT* iterations
        step_size: Maximum step size for extending the tree (in meters)
        goal_sample_rate: Probability of sampling the goal
        search_radius: Radius for rewiring in RRT* (in meters)
        goal_threshold: Distance threshold to consider goal reached (in meters)
        collision_check_step: Step size for collision checking along the path
        workspace_limits: Limits of the workspace in Cartesian space [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
    """
    def __init__(
        self,
        robot_id: int,
        joint_indices: List[int],
        lower_limits: List[float],
        upper_limits: List[float],
        ee_link_index: int,
        obstacle_tracker: Any,
        max_iterations: int = 1000,
        step_size: float = 0.05,
        goal_sample_rate: float = 0.05,
        search_radius: float = 0.1,
        goal_threshold: float = 0.03,
        collision_check_step: float = 0.02,
        workspace_limits: List[List[float]] = None
    ):
        self.robot_id = robot_id
        self.joint_indices = joint_indices
        self.lower_limits = lower_limits
        self.upper_limits = upper_limits
        self.ee_link_index = ee_link_index
        self.obstacle_tracker = obstacle_tracker
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.search_radius = search_radius
        self.goal_threshold = goal_threshold
        self.collision_check_step = collision_check_step
        
        # Default workspace limits if not provided
        if workspace_limits is None:
            # Set a default workspace range
            self.workspace_limits = [
                [0.2, 0.8],  # x_min, x_max
                [-0.5, 0.5],  # y_min, y_max
                [0.0, 0.8]   # z_min, z_max
            ]
        else:
            self.workspace_limits = workspace_limits
        
        # Initialize IK solver
        self.ik_solver = DifferentialIKSolver(robot_id, ee_link_index, damping=0.05)
        
        # Tree structure
        self.nodes_cart = []  # Nodes in Cartesian space
        self.nodes_joint = []  # Corresponding joint space positions
        self.costs = []  # Cost from start to each node
        self.parents = []  # Parent node index for each node
        
        # Visualization
        self.debug_lines = []
        
    def _get_current_ee_pose(self, joint_positions: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Get end effector pose for given joint positions.
        
        Args:
            joint_positions: Joint positions
            
        Returns:
            Tuple of end effector position and orientation
        """
        # Save current state
        current_states = []
        for i in self.joint_indices:
            current_states.append(p.getJointState(self.robot_id, i)[0])
            
        # Set joint positions
        for i, idx in enumerate(self.joint_indices):
            p.resetJointState(self.robot_id, idx, joint_positions[i])
            
        # Get end effector pose
        ee_state = p.getLinkState(self.robot_id, self.ee_link_index)
        ee_pos = np.array(ee_state[0])
        ee_orn = np.array(ee_state[1])
        
        # Restore original state
        for i, idx in enumerate(self.joint_indices):
            p.resetJointState(self.robot_id, idx, current_states[i])
            
        return ee_pos, ee_orn
    
    def _is_collision_free_joint(self, start_joints: List[float], end_joints: List[float]) -> bool:
        """Check if path between two joint configurations is collision-free.
        
        Args:
            start_joints: Starting joint configuration
            end_joints: Ending joint configuration
            
        Returns:
            True if path is collision-free, False otherwise
        """
        # Get distance in joint space
        dist = np.linalg.norm(np.array(end_joints) - np.array(start_joints))
        
        # Number of steps for collision checking
        n_steps = max(2, int(dist / self.collision_check_step))
        
        # Check each step along the path
        for i in range(n_steps + 1):
            t = i / n_steps
            # Linear interpolation
            joint_pos = [start + t * (end - start) for start, end in zip(start_joints, end_joints)]
            
            # Check height constraint
            if not self._is_ee_height_valid(joint_pos):
                return False
                
            # Check collision with obstacles
            if self._is_state_in_collision(joint_pos):
                return False
                
        return True
    
    def _is_collision_free_cart(self, start_pos: np.ndarray, end_pos: np.ndarray, 
                               start_joints: List[float], end_joints: List[float]) -> bool:
        """Check if path between two points in Cartesian space is collision-free.
        
        Args:
            start_pos: Starting position
            end_pos: Ending position
            start_joints: Starting joint configuration
            end_joints: Ending joint configuration
            
        Returns:
            True if path is collision-free, False otherwise
        """
        # Get distance in Cartesian space
        dist = np.linalg.norm(end_pos - start_pos)
        
        # Number of steps for collision checking
        n_steps = max(2, int(dist / self.collision_check_step))
        
        # Get current end effector orientation
        _, start_orn = self._get_current_ee_pose(start_joints)
        
        # Check each step along the path
        for i in range(n_steps + 1):
            t = i / n_steps
            # Linear interpolation
            pos = start_pos + t * (end_pos - start_pos)
            
            # Get IK solution for current position
            if i == 0:
                joint_pos = start_joints
            elif i == n_steps:
                joint_pos = end_joints
            else:
                # Solve IK for current Cartesian position
                try:
                    # Use linearly interpolated joint position as initial guess
                    init_guess = [start + t * (end - start) for start, end in zip(start_joints, end_joints)]
                    joint_pos = self.ik_solver.solve(pos, start_orn, init_guess, max_iters=20, tolerance=0.005)
                except:
                    # IK solution failed
                    return False
            
            # Check height constraint
            if not self._is_ee_height_valid(joint_pos):
                return False
                
            # Check collision with obstacles
            if self._is_state_in_collision(joint_pos):
                return False
                
        return True
    
    def _is_state_in_collision(self, joint_pos: List[float]) -> bool:
        """Check if joint state is in collision with obstacles.
        
        Args:
            joint_pos: Joint positions to check
            
        Returns:
            True if in collision, False otherwise
        """
        # Get end effector position and orientation
        ee_pos, _ = self._get_current_ee_pose(joint_pos)
        
        # Get robot link positions for collision checking
        # We'll check several key links along the robot chain
        links_to_check = self.joint_indices + [self.ee_link_index]
        
        # Save current state
        current_states = []
        for i in self.joint_indices:
            current_states.append(p.getJointState(self.robot_id, i)[0])
            
        # Set joint positions
        for i, idx in enumerate(self.joint_indices):
            p.resetJointState(self.robot_id, idx, joint_pos[i])
        
        # Check collision with obstacles
        collision = False
        
        # Get obstacle states from tracker
        obstacle_states = self.obstacle_tracker.get_all_obstacle_states()
        if obstacle_states is None or len(obstacle_states) == 0:
            # Restore original state
            for i, idx in enumerate(self.joint_indices):
                p.resetJointState(self.robot_id, idx, current_states[i])
            return False
        
        # Check each link against each obstacle
        for link_idx in links_to_check:
            link_state = p.getLinkState(self.robot_id, link_idx)
            link_pos = np.array(link_state[0])
            
            for obstacle in obstacle_states:
                if obstacle is None:
                    continue
                
                # Simple sphere collision check
                obstacle_pos = obstacle['position']
                obstacle_radius = obstacle['radius']
                
                # Distance between link and obstacle center
                dist = np.linalg.norm(link_pos - obstacle_pos)
                
                # Approximate robot link as a point (simplified)
                # Add a small safety margin (0.05 meters)
                if dist < obstacle_radius + 0.05:
                    collision = True
                    break
            
            if collision:
                break
                
        # Restore original state
        for i, idx in enumerate(self.joint_indices):
            p.resetJointState(self.robot_id, idx, current_states[i])
            
        return collision
    
    def _distance_cart(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Calculate distance between two points in Cartesian space.
        
        Args:
            p1: First point
            p2: Second point
            
        Returns:
            Euclidean distance in Cartesian space
        """
        return np.linalg.norm(p1 - p2)
    
    def _sample_random_cart_point(self) -> np.ndarray:
        """Sample a random point in Cartesian space.
        
        Returns:
            Random Cartesian coordinates
        """
        # Sample within workspace limits
        x = random.uniform(self.workspace_limits[0][0], self.workspace_limits[0][1])
        y = random.uniform(self.workspace_limits[1][0], self.workspace_limits[1][1])
        z = random.uniform(self.workspace_limits[2][0], self.workspace_limits[2][1])
        
        return np.array([x, y, z])
    
    def _is_ee_height_valid(self, joint_pos: List[float]) -> bool:
        """Check if end effector height is valid (above table).
        
        Args:
            joint_pos: Joint positions to check
            
        Returns:
            True if end effector height is valid, False otherwise
        """
        # Get end effector position
        ee_pos, _ = self._get_current_ee_pose(joint_pos)
        
        # Get robot base position (assuming index 0)
        # We can access the robot base position or use a fixed threshold for table height
        # Here, we use a simple approach to check if ee_pos[2] (z-coordinate) is above a threshold
        
        # Get base link position
        base_pos = p.getBasePositionAndOrientation(self.robot_id)[0]
        table_height = base_pos[2]  # Base Z coordinate represents table height
        
        # Add a small threshold to account for base height itself
        min_height = table_height - 0.01  # 1cm margin below base
        
        # Check if end effector is above table height
        return ee_pos[2] > min_height
    
    def _steer(self, from_point: np.ndarray, to_point: np.ndarray) -> np.ndarray:
        """Steer from one point toward another in Cartesian space.
        
        If points are closer than step_size, returns target point directly.
        Otherwise, moves step_size distance in the direction of target.
        
        Args:
            from_point: Starting point
            to_point: Target point
            
        Returns:
            New point
        """
        dist = self._distance_cart(from_point, to_point)
        
        if dist < self.step_size:
            return to_point
        else:
            # Calculate direction vector
            dir_vec = (to_point - from_point) / dist
            # Move by step size
            return from_point + dir_vec * self.step_size
    
    def _calculate_cost(self, node_idx: int) -> float:
        """Calculate total cost from start to given node.
        
        Args:
            node_idx: Index of node to calculate cost for
            
        Returns:
            Total cost from start to node
        """
        cost = 0.0
        current_idx = node_idx
        
        while current_idx != 0:  # Until we reach the root node (index 0)
            parent_idx = self.parents[current_idx]
            cost += self._distance_cart(self.nodes_cart[current_idx], self.nodes_cart[parent_idx])
            current_idx = parent_idx
            
        return cost
    
    def _choose_parent(self, new_point: np.ndarray, nearby_indices: List[int], new_joint_state: List[float]) -> Tuple[int, float]:
        """Choose best parent for a new node.
        
        Selects parent that results in minimum total cost.
        
        Args:
            new_point: Cartesian position of new node
            nearby_indices: List of indices of nearby nodes
            new_joint_state: Joint state of new node
            
        Returns:
            (Best parent index, cost of new node)
        """
        if not nearby_indices:
            return -1, float('inf')
            
        costs = []
        valid_indices = []
        
        for idx in nearby_indices:
            # Check if we can connect to this node
            if self._is_collision_free_cart(
                self.nodes_cart[idx], 
                new_point, 
                self.nodes_joint[idx], 
                new_joint_state
            ):
                # Calculate cost through this parent
                cost = self.costs[idx] + self._distance_cart(self.nodes_cart[idx], new_point)
                costs.append(cost)
                valid_indices.append(idx)
                
        if not valid_indices:
            return -1, float('inf')
            
        # Find index of minimum cost
        min_cost_idx = np.argmin(costs)
        return valid_indices[min_cost_idx], costs[min_cost_idx]
    
    def _rewire(self, new_node_idx: int, nearby_indices: List[int]) -> None:
        """Check if nearby nodes can have lower cost by connecting through new node.
        
        Args:
            new_node_idx: Index of newly added node
            nearby_indices: List of indices of nearby nodes
        """
        for idx in nearby_indices:
            # Skip parent
            if idx == self.parents[new_node_idx]:
                continue
                
            # Check if we can connect to this node
            if self._is_collision_free_cart(
                self.nodes_cart[new_node_idx], 
                self.nodes_cart[idx], 
                self.nodes_joint[new_node_idx], 
                self.nodes_joint[idx]
            ):
                # Calculate new cost
                new_cost = self.costs[new_node_idx] + self._distance_cart(self.nodes_cart[new_node_idx], self.nodes_cart[idx])
                
                # If new cost is lower, rewire
                if new_cost < self.costs[idx]:
                    self.parents[idx] = new_node_idx
                    self.costs[idx] = new_cost
                    # Update visualization
                    self._update_visualization(idx)
    
    def _find_nearby(self, point: np.ndarray) -> List[int]:
        """Find all nodes near a given point in Cartesian space.
        
        Args:
            point: Query point
            
        Returns:
            List of indices of nodes within search_radius
        """
        nearby_indices = []
        
        for i, node in enumerate(self.nodes_cart):
            if self._distance_cart(point, node) < self.search_radius:
                nearby_indices.append(i)
                
        return nearby_indices
    
    def _update_visualization(self, node_idx: int) -> None:
        """Update visualization of connection between node and its parent.
        
        Args:
            node_idx: Index of node to update visualization for
        """
        # Remove old lines
        for line_id in self.debug_lines:
            p.removeUserDebugItem(line_id)
        self.debug_lines = []
        
        # Create new lines to visualize tree
        for i in range(1, len(self.nodes_cart)):
            parent_idx = self.parents[i]
            start_pos = self.nodes_cart[parent_idx]
            end_pos = self.nodes_cart[i]
            
            line_id = p.addUserDebugLine(
                start_pos.tolist(),
                end_pos.tolist(),
                lineColorRGB=[0, 0.8, 0.2],  # Green line
                lineWidth=1
            )
            self.debug_lines.append(line_id)
    
    def plan(self, start_joint_config: List[float], goal_ee_pos: np.ndarray, goal_ee_orn: np.ndarray) -> Tuple[List[List[float]], float]:
        """Plan path from start joint configuration to goal end effector position.
        
        Args:
            start_joint_config: Starting joint configuration
            goal_ee_pos: Goal end effector position
            goal_ee_orn: Goal end effector orientation
            
        Returns:
            (path, path_cost)
            path is a list of joint positions from start to goal
        """
        # Reset tree
        self.nodes_cart = []
        self.nodes_joint = []
        self.costs = []
        self.parents = []
        
        # Get end effector pose at starting position
        start_ee_pos, start_ee_orn = self._get_current_ee_pose(start_joint_config)
        
        # Try to get IK solution for goal position
        try:
            goal_joint_config = self.ik_solver.solve(goal_ee_pos, goal_ee_orn, start_joint_config, max_iters=50, tolerance=0.001)
        except:
            print("Cannot find IK solution for goal position")
            return [], float('inf')
        
        # Initialize tree
        self.nodes_cart.append(start_ee_pos)
        self.nodes_joint.append(start_joint_config)
        self.costs.append(0.0)
        self.parents.append(0)  # Root node is its own parent
        
        # Track node closest to goal
        best_goal_idx = 0
        best_goal_distance = self._distance_cart(start_ee_pos, goal_ee_pos)
        
        # RRT* main loop
        for i in range(self.max_iterations):
            # Sample goal point with certain probability
            if random.random() < self.goal_sample_rate:
                sample_point = goal_ee_pos
            else:
                sample_point = self._sample_random_cart_point()
            
            # Find nearest node to sampled point
            distances = [self._distance_cart(sample_point, node) for node in self.nodes_cart]
            nearest_idx = np.argmin(distances)
            
            # Steer toward sampled point
            new_point = self._steer(self.nodes_cart[nearest_idx], sample_point)
            
            # Use nearest node's joint state as initial guess for IK
            try:
                new_joint_state = self.ik_solver.solve(
                    new_point, 
                    start_ee_orn,  # Maintain initial orientation 
                    self.nodes_joint[nearest_idx], 
                    max_iters=20, 
                    tolerance=0.005
                )
            except:
                # IK solution failed, skip this point
                continue
            
            # Check if new point is collision-free
            if not self._is_collision_free_cart(
                self.nodes_cart[nearest_idx], 
                new_point, 
                self.nodes_joint[nearest_idx], 
                new_joint_state
            ):
                continue
            
            # Find nearby nodes
            nearby_indices = self._find_nearby(new_point)
            
            # Choose best parent
            best_parent_idx, new_cost = self._choose_parent(new_point, nearby_indices, new_joint_state)
            
            if best_parent_idx == -1:
                # No valid parent
                continue
            
            # Add new node
            new_node_idx = len(self.nodes_cart)
            self.nodes_cart.append(new_point)
            self.nodes_joint.append(new_joint_state)
            self.costs.append(new_cost)
            self.parents.append(best_parent_idx)
            
            # Rewire
            self._rewire(new_node_idx, nearby_indices)
            
            # Update visualization
            self._update_visualization(new_node_idx)
            
            # Check if new node is closer to goal
            distance_to_goal = self._distance_cart(new_point, goal_ee_pos)
            if distance_to_goal < best_goal_distance:
                best_goal_distance = distance_to_goal
                best_goal_idx = new_node_idx
                
                # Print current best distance
                if i % 10 == 0:
                    print(f"Iteration {i}: Current closest distance to goal = {best_goal_distance:.6f}")
            
            # Check if goal is reached
            if distance_to_goal < self.goal_threshold:
                print(f"Goal reached! Iterations: {i}, Distance: {distance_to_goal:.6f}")
                
                # Connect directly to actual goal
                goal_node_idx = len(self.nodes_cart)
                self.nodes_cart.append(goal_ee_pos)
                self.nodes_joint.append(goal_joint_config)
                goal_cost = self.costs[new_node_idx] + self._distance_cart(new_point, goal_ee_pos)
                self.costs.append(goal_cost)
                self.parents.append(new_node_idx)
                
                # Extract path
                path = self._extract_path(goal_node_idx)
                return path, goal_cost
        
        # If max iterations reached but no path found
        print(f"Max iterations reached ({self.max_iterations}), returning best path to goal")
        print(f"Best distance: {best_goal_distance:.6f}")
        
        # If we have at least one node close to goal
        if best_goal_distance < 0.1:  # 10cm threshold
            # Try to connect to actual goal
            if self._is_collision_free_cart(
                self.nodes_cart[best_goal_idx], 
                goal_ee_pos, 
                self.nodes_joint[best_goal_idx], 
                goal_joint_config
            ):
                print("Connecting to actual goal")
                goal_node_idx = len(self.nodes_cart)
                self.nodes_cart.append(goal_ee_pos)
                self.nodes_joint.append(goal_joint_config)
                goal_cost = self.costs[best_goal_idx] + self._distance_cart(self.nodes_cart[best_goal_idx], goal_ee_pos)
                self.costs.append(goal_cost)
                self.parents.append(best_goal_idx)
                
                # Extract path
                path = self._extract_path(goal_node_idx)
                return path, goal_cost
        
        # Extract path to closest node
        path = self._extract_path(best_goal_idx)
        return path, self.costs[best_goal_idx]
    
    def _extract_path(self, goal_idx: int) -> List[List[float]]:
        """Extract path from tree.
        
        Args:
            goal_idx: Index of goal node
            
        Returns:
            List of joint positions from start to goal
        """
        path = []
        current_idx = goal_idx
        
        # Trace path from goal to start
        while current_idx != 0:
            path.append(self.nodes_joint[current_idx])
            current_idx = self.parents[current_idx]
            
        # Add start point
        path.append(self.nodes_joint[0])
        
        # Reverse path to get start to goal
        path.reverse()
        
        return path
    
    def clear_visualization(self) -> None:
        """Clear all visualization elements."""
        for line_id in self.debug_lines:
            p.removeUserDebugItem(line_id)
        self.debug_lines = []
    
    def generate_smooth_trajectory(self, path: List[List[float]], smoothing_steps: int = 10) -> List[List[float]]:
        """Generate smooth trajectory.
        
        Args:
            path: Original path (list of joint positions)
            smoothing_steps: Number of steps to insert between path points
            
        Returns:
            Smooth trajectory
        """
        if not path or len(path) < 2:
            return path
            
        smooth_path = []
        
        for i in range(len(path) - 1):
            start_config = path[i]
            end_config = path[i+1]
            
            for step in range(smoothing_steps):
                t = step / smoothing_steps
                # Linear interpolation
                config = [start + t * (end - start) for start, end in zip(start_config, end_config)]
                smooth_path.append(config)
                
        # Add final configuration
        smooth_path.append(path[-1])
        
        return smooth_path


# Test function
def test_rrt_star_cartesian():
    """Test Cartesian space RRT* planner"""
    import pybullet as p
    import pybullet_data
    import time

    # Initialize PyBullet
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    
    # Load robot and environment
    plane_id = p.loadURDF("plane.urdf")
    robot_id = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)
    
    # Set robot initial state
    for i in range(p.getNumJoints(robot_id)):
        p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, 0)
    
    # Get joint information
    joint_indices = []
    lower_limits = []
    upper_limits = []
    
    for i in range(p.getNumJoints(robot_id)):
        joint_info = p.getJointInfo(robot_id, i)
        if joint_info[2] == p.JOINT_REVOLUTE:
            joint_indices.append(i)
            lower_limits.append(joint_info[8])
            upper_limits.append(joint_info[9])
    
    # End effector index
    ee_link_index = 11  # Assuming end effector link index is 11, adjust based on actual robot model
    
    # Create simple obstacle tracker (simulation)
    class SimpleObstacleTracker:
        def __init__(self):
            self.obstacles = []
            # Add some obstacles
            self.add_obstacle([0.5, 0.3, 0.2], 0.1)
            self.add_obstacle([0.5, -0.3, 0.2], 0.1)
            
        def add_obstacle(self, position, radius):
            visual_id = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=[1, 0, 0, 0.7])
            body_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_id, basePosition=position)
            self.obstacles.append({
                'id': body_id,
                'position': np.array(position),
                'radius': radius
            })
            
        def get_all_obstacle_states(self):
            return self.obstacles
    
    obstacle_tracker = SimpleObstacleTracker()
    
    # Create planner
    planner = RRTStarCartesianPlanner(
        robot_id=robot_id,
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
    
    # Define starting joint configuration
    start_config = [0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0]
    
    # Define goal position and orientation
    goal_pos = np.array([0.6, 0.2, 0.5])
    goal_orn = p.getQuaternionFromEuler([0, -np.pi/2, 0])
    
    # Visualize goal position
    p.addUserDebugPoints([goal_pos], [[1, 0, 0]], pointSize=10)
    
    # Plan path
    print("Starting path planning...")
    path, cost = planner.plan(start_config, goal_pos, goal_orn)
    
    if path:
        print(f"Path found! Cost: {cost:.6f}")
        
        # Generate smooth trajectory
        smooth_path = planner.generate_smooth_trajectory(path, smoothing_steps=20)
        
        # Execute trajectory
        print("Executing trajectory...")
        for joint_pos in smooth_path:
            # Set joint positions
            for i, idx in enumerate(joint_indices):
                p.setJointMotorControl2(robot_id, idx, p.POSITION_CONTROL, joint_pos[i])
            
            # Update simulation
            p.stepSimulation()
            time.sleep(0.01)
    else:
        print("No path found")
    
    # Keep window open
    while p.isConnected():
        p.stepSimulation()
        time.sleep(0.01)
    

if __name__ == "__main__":
    test_rrt_star_cartesian() 