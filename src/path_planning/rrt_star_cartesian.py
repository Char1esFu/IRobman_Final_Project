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
        
        # 默认工作空间限制，如果没有提供
        if workspace_limits is None:
            # 设置一个默认的工作空间范围
            self.workspace_limits = [
                [0.2, 0.8],  # x_min, x_max
                [-0.5, 0.5],  # y_min, y_max
                [0.0, 0.8]   # z_min, z_max
            ]
        else:
            self.workspace_limits = workspace_limits
        
        # 初始化IK求解器
        self.ik_solver = DifferentialIKSolver(robot_id, ee_link_index, damping=0.05)
        
        # 树的结构
        self.nodes_cart = []  # 笛卡尔空间中的节点
        self.nodes_joint = []  # 对应的关节空间位置
        self.costs = []  # 从起点到每个节点的代价
        self.parents = []  # 每个节点的父节点索引
        
        # 可视化
        self.debug_lines = []
        
    def _get_current_ee_pose(self, joint_positions: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """获取给定关节位置的末端执行器姿态。
        
        Args:
            joint_positions: 关节位置
            
        Returns:
            末端执行器位置和方向的元组
        """
        # 保存当前状态
        current_states = []
        for i in self.joint_indices:
            current_states.append(p.getJointState(self.robot_id, i)[0])
            
        # 设置关节位置
        for i, idx in enumerate(self.joint_indices):
            p.resetJointState(self.robot_id, idx, joint_positions[i])
            
        # 获取末端执行器姿态
        ee_state = p.getLinkState(self.robot_id, self.ee_link_index)
        ee_pos = np.array(ee_state[0])
        ee_orn = np.array(ee_state[1])
        
        # 恢复原始状态
        for i, idx in enumerate(self.joint_indices):
            p.resetJointState(self.robot_id, idx, current_states[i])
            
        return ee_pos, ee_orn
    
    def _is_collision_free_joint(self, start_joints: List[float], end_joints: List[float]) -> bool:
        """检查两个关节配置之间的路径是否无碰撞。
        
        Args:
            start_joints: 起始关节配置
            end_joints: 结束关节配置
            
        Returns:
            如果路径无碰撞则为True，否则为False
        """
        # 获取关节空间中的距离
        dist = np.linalg.norm(np.array(end_joints) - np.array(start_joints))
        
        # 碰撞检查的步数
        n_steps = max(2, int(dist / self.collision_check_step))
        
        # 检查路径上的每一步
        for i in range(n_steps + 1):
            t = i / n_steps
            # 线性插值
            joint_pos = [start + t * (end - start) for start, end in zip(start_joints, end_joints)]
            
            # 检查高度约束
            if not self._is_ee_height_valid(joint_pos):
                return False
                
            # 检查与障碍物的碰撞
            if self._is_state_in_collision(joint_pos):
                return False
                
        return True
    
    def _is_collision_free_cart(self, start_pos: np.ndarray, end_pos: np.ndarray, 
                               start_joints: List[float], end_joints: List[float]) -> bool:
        """检查笛卡尔空间中两点之间的路径是否无碰撞。
        
        Args:
            start_pos: 起始位置
            end_pos: 结束位置
            start_joints: 起始关节配置
            end_joints: 结束关节配置
            
        Returns:
            如果路径无碰撞则为True，否则为False
        """
        # 获取笛卡尔空间中的距离
        dist = np.linalg.norm(end_pos - start_pos)
        
        # 碰撞检查的步数
        n_steps = max(2, int(dist / self.collision_check_step))
        
        # 获取当前末端执行器姿态
        _, start_orn = self._get_current_ee_pose(start_joints)
        
        # 检查路径上的每一步
        for i in range(n_steps + 1):
            t = i / n_steps
            # 线性插值
            pos = start_pos + t * (end_pos - start_pos)
            
            # 获取当前位置的IK解
            if i == 0:
                joint_pos = start_joints
            elif i == n_steps:
                joint_pos = end_joints
            else:
                # 解算当前笛卡尔位置的IK
                try:
                    # 使用线性插值的关节位置作为初始猜测
                    init_guess = [start + t * (end - start) for start, end in zip(start_joints, end_joints)]
                    joint_pos = self.ik_solver.solve(pos, start_orn, init_guess, max_iters=20, tolerance=0.005)
                except:
                    # IK求解失败
                    return False
            
            # 检查高度约束
            if not self._is_ee_height_valid(joint_pos):
                return False
                
            # 检查与障碍物的碰撞
            if self._is_state_in_collision(joint_pos):
                return False
                
        return True
    
    def _is_state_in_collision(self, joint_pos: List[float]) -> bool:
        """检查关节状态是否与障碍物碰撞。
        
        Args:
            joint_pos: 要检查的关节位置
            
        Returns:
            如果碰撞则为True，否则为False
        """
        # 获取末端执行器位置和方向
        ee_pos, _ = self._get_current_ee_pose(joint_pos)
        
        # 获取用于碰撞检查的机器人链接位置
        # 我们将检查机器人运动链上的几个关键链接
        links_to_check = self.joint_indices + [self.ee_link_index]
        
        # 保存当前状态
        current_states = []
        for i in self.joint_indices:
            current_states.append(p.getJointState(self.robot_id, i)[0])
            
        # 设置关节位置
        for i, idx in enumerate(self.joint_indices):
            p.resetJointState(self.robot_id, idx, joint_pos[i])
        
        # 检查与障碍物的碰撞
        collision = False
        
        # 从跟踪器获取障碍物状态
        obstacle_states = self.obstacle_tracker.get_all_obstacle_states()
        if obstacle_states is None or len(obstacle_states) == 0:
            # 恢复原始状态
            for i, idx in enumerate(self.joint_indices):
                p.resetJointState(self.robot_id, idx, current_states[i])
            return False
        
        # 检查每个链接与每个障碍物
        for link_idx in links_to_check:
            link_state = p.getLinkState(self.robot_id, link_idx)
            link_pos = np.array(link_state[0])
            
            for obstacle in obstacle_states:
                if obstacle is None:
                    continue
                
                # 简单的球体碰撞检查
                obstacle_pos = obstacle['position']
                obstacle_radius = obstacle['radius']
                
                # 链接与障碍物中心之间的距离
                dist = np.linalg.norm(link_pos - obstacle_pos)
                
                # 将机器人链接近似为一个点（简化）
                # 增加一个小的安全边距（0.05米）
                if dist < obstacle_radius + 0.05:
                    collision = True
                    break
            
            if collision:
                break
                
        # 恢复原始状态
        for i, idx in enumerate(self.joint_indices):
            p.resetJointState(self.robot_id, idx, current_states[i])
            
        return collision
    
    def _distance_cart(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """计算笛卡尔空间中两点之间的距离。
        
        Args:
            p1: 第一个点
            p2: 第二个点
            
        Returns:
            笛卡尔空间中的欧几里得距离
        """
        return np.linalg.norm(p1 - p2)
    
    def _sample_random_cart_point(self) -> np.ndarray:
        """在笛卡尔空间中采样随机点。
        
        Returns:
            随机的笛卡尔坐标
        """
        # 在工作空间限制内采样
        x = random.uniform(self.workspace_limits[0][0], self.workspace_limits[0][1])
        y = random.uniform(self.workspace_limits[1][0], self.workspace_limits[1][1])
        z = random.uniform(self.workspace_limits[2][0], self.workspace_limits[2][1])
        
        return np.array([x, y, z])
    
    def _is_ee_height_valid(self, joint_pos: List[float]) -> bool:
        """检查末端执行器高度是否有效（高于桌面）。
        
        Args:
            joint_pos: 要检查的关节位置
            
        Returns:
            如果末端执行器高度有效则为True，否则为False
        """
        # 获取末端执行器位置
        ee_pos, _ = self._get_current_ee_pose(joint_pos)
        
        # 获取机器人基座位置（假设索引为0）
        # 我们可以访问机器人基座位置或使用固定阈值表示桌面高度
        # 这里，我们使用一种简单的方法来检查ee_pos[2]（z坐标）是否高于阈值
        
        # 获取基座链接位置
        base_pos = p.getBasePositionAndOrientation(self.robot_id)[0]
        table_height = base_pos[2]  # 基座Z坐标表示桌面高度
        
        # 添加一个小阈值来考虑基座高度本身
        min_height = table_height - 0.01  # 基座下方1厘米边距
        
        # 检查末端执行器是否高于桌面高度
        return ee_pos[2] > min_height
    
    def _steer(self, from_point: np.ndarray, to_point: np.ndarray) -> np.ndarray:
        """在笛卡尔空间中从一个点向另一个点引导。
        
        如果点距离小于步长，则直接返回目标点。
        否则，向目标方向移动step_size距离。
        
        Args:
            from_point: 起始点
            to_point: 目标点
            
        Returns:
            新点
        """
        dist = self._distance_cart(from_point, to_point)
        
        if dist < self.step_size:
            return to_point
        else:
            # 计算方向向量
            dir_vec = (to_point - from_point) / dist
            # 按步长移动
            return from_point + dir_vec * self.step_size
    
    def _calculate_cost(self, node_idx: int) -> float:
        """计算从起点到给定节点的总代价。
        
        Args:
            node_idx: 要计算代价的节点索引
            
        Returns:
            从起点到该节点的总代价
        """
        cost = 0.0
        current_idx = node_idx
        
        while current_idx != 0:  # 直到到达根节点（索引0）
            parent_idx = self.parents[current_idx]
            cost += self._distance_cart(self.nodes_cart[current_idx], self.nodes_cart[parent_idx])
            current_idx = parent_idx
            
        return cost
    
    def _choose_parent(self, new_point: np.ndarray, nearby_indices: List[int], new_joint_state: List[float]) -> Tuple[int, float]:
        """为新节点选择最佳父节点。
        
        选择会产生最小总代价的父节点。
        
        Args:
            new_point: 新节点的笛卡尔位置
            nearby_indices: 附近节点的索引列表
            new_joint_state: 新节点的关节状态
            
        Returns:
            (最佳父节点索引, 新节点的代价)
        """
        if not nearby_indices:
            return -1, float('inf')
            
        costs = []
        valid_indices = []
        
        for idx in nearby_indices:
            # 检查是否可以连接到此节点
            if self._is_collision_free_cart(
                self.nodes_cart[idx], 
                new_point, 
                self.nodes_joint[idx], 
                new_joint_state
            ):
                # 计算通过此父节点的代价
                cost = self.costs[idx] + self._distance_cart(self.nodes_cart[idx], new_point)
                costs.append(cost)
                valid_indices.append(idx)
                
        if not valid_indices:
            return -1, float('inf')
            
        # 找到最小代价的索引
        min_cost_idx = np.argmin(costs)
        return valid_indices[min_cost_idx], costs[min_cost_idx]
    
    def _rewire(self, new_node_idx: int, nearby_indices: List[int]) -> None:
        """检查是否可以通过新节点降低附近节点的代价。
        
        Args:
            new_node_idx: 新添加的节点索引
            nearby_indices: 附近节点的索引列表
        """
        for idx in nearby_indices:
            # 跳过父节点
            if idx == self.parents[new_node_idx]:
                continue
                
            # 检查是否可以连接到此节点
            if self._is_collision_free_cart(
                self.nodes_cart[new_node_idx], 
                self.nodes_cart[idx], 
                self.nodes_joint[new_node_idx], 
                self.nodes_joint[idx]
            ):
                # 计算新代价
                new_cost = self.costs[new_node_idx] + self._distance_cart(self.nodes_cart[new_node_idx], self.nodes_cart[idx])
                
                # 如果新代价更低，则重新连接
                if new_cost < self.costs[idx]:
                    self.parents[idx] = new_node_idx
                    self.costs[idx] = new_cost
                    # 可视化更新
                    self._update_visualization(idx)
    
    def _find_nearby(self, point: np.ndarray) -> List[int]:
        """找到笛卡尔空间中给定点附近的所有节点。
        
        Args:
            point: 查询点
            
        Returns:
            距离小于search_radius的节点索引列表
        """
        nearby_indices = []
        
        for i, node in enumerate(self.nodes_cart):
            if self._distance_cart(point, node) < self.search_radius:
                nearby_indices.append(i)
                
        return nearby_indices
    
    def _update_visualization(self, node_idx: int) -> None:
        """更新节点及其父节点之间连接的可视化。
        
        Args:
            node_idx: 要更新可视化的节点索引
        """
        # 移除旧线
        for line_id in self.debug_lines:
            p.removeUserDebugItem(line_id)
        self.debug_lines = []
        
        # 创建新线以可视化树
        for i in range(1, len(self.nodes_cart)):
            parent_idx = self.parents[i]
            start_pos = self.nodes_cart[parent_idx]
            end_pos = self.nodes_cart[i]
            
            line_id = p.addUserDebugLine(
                start_pos.tolist(),
                end_pos.tolist(),
                lineColorRGB=[0, 0.8, 0.2],  # 绿色线
                lineWidth=1
            )
            self.debug_lines.append(line_id)
    
    def plan(self, start_joint_config: List[float], goal_ee_pos: np.ndarray, goal_ee_orn: np.ndarray) -> Tuple[List[List[float]], float]:
        """规划从起始关节配置到目标末端执行器位置的路径。
        
        Args:
            start_joint_config: 起始关节配置
            goal_ee_pos: 目标末端执行器位置
            goal_ee_orn: 目标末端执行器方向
            
        Returns:
            (路径, 路径代价)
            路径是关节位置列表，从起点到终点
        """
        # 重置树
        self.nodes_cart = []
        self.nodes_joint = []
        self.costs = []
        self.parents = []
        
        # 获取起始位置的末端执行器姿态
        start_ee_pos, start_ee_orn = self._get_current_ee_pose(start_joint_config)
        
        # 尝试获取目标位置的IK解
        try:
            goal_joint_config = self.ik_solver.solve(goal_ee_pos, goal_ee_orn, start_joint_config, max_iters=50, tolerance=0.001)
        except:
            print("无法为目标位置找到IK解")
            return [], float('inf')
        
        # 初始化树
        self.nodes_cart.append(start_ee_pos)
        self.nodes_joint.append(start_joint_config)
        self.costs.append(0.0)
        self.parents.append(0)  # 根节点是自己的父节点
        
        # 记录最接近目标的节点
        best_goal_idx = 0
        best_goal_distance = self._distance_cart(start_ee_pos, goal_ee_pos)
        
        # RRT* 主循环
        for i in range(self.max_iterations):
            # 以一定概率直接采样目标点
            if random.random() < self.goal_sample_rate:
                sample_point = goal_ee_pos
            else:
                sample_point = self._sample_random_cart_point()
            
            # 找到离采样点最近的节点
            distances = [self._distance_cart(sample_point, node) for node in self.nodes_cart]
            nearest_idx = np.argmin(distances)
            
            # 向采样点引导
            new_point = self._steer(self.nodes_cart[nearest_idx], sample_point)
            
            # 使用最近节点的关节状态作为IK的初始猜测
            try:
                new_joint_state = self.ik_solver.solve(
                    new_point, 
                    start_ee_orn,  # 保持初始方向 
                    self.nodes_joint[nearest_idx], 
                    max_iters=20, 
                    tolerance=0.005
                )
            except:
                # IK求解失败，跳过此点
                continue
            
            # 检查新点是否无碰撞
            if not self._is_collision_free_cart(
                self.nodes_cart[nearest_idx], 
                new_point, 
                self.nodes_joint[nearest_idx], 
                new_joint_state
            ):
                continue
            
            # 找到附近的节点
            nearby_indices = self._find_nearby(new_point)
            
            # 选择最佳父节点
            best_parent_idx, new_cost = self._choose_parent(new_point, nearby_indices, new_joint_state)
            
            if best_parent_idx == -1:
                # 没有有效的父节点
                continue
            
            # 添加新节点
            new_node_idx = len(self.nodes_cart)
            self.nodes_cart.append(new_point)
            self.nodes_joint.append(new_joint_state)
            self.costs.append(new_cost)
            self.parents.append(best_parent_idx)
            
            # 重新连接
            self._rewire(new_node_idx, nearby_indices)
            
            # 更新可视化
            self._update_visualization(new_node_idx)
            
            # 检查新节点是否更接近目标
            distance_to_goal = self._distance_cart(new_point, goal_ee_pos)
            if distance_to_goal < best_goal_distance:
                best_goal_distance = distance_to_goal
                best_goal_idx = new_node_idx
                
                # 打印当前最佳距离
                if i % 10 == 0:
                    print(f"迭代 {i}: 当前离目标最近的距离 = {best_goal_distance:.6f}")
            
            # 检查是否达到目标
            if distance_to_goal < self.goal_threshold:
                print(f"到达目标! 迭代次数: {i}, 距离: {distance_to_goal:.6f}")
                
                # 直接连接到实际目标
                goal_node_idx = len(self.nodes_cart)
                self.nodes_cart.append(goal_ee_pos)
                self.nodes_joint.append(goal_joint_config)
                goal_cost = self.costs[new_node_idx] + self._distance_cart(new_point, goal_ee_pos)
                self.costs.append(goal_cost)
                self.parents.append(new_node_idx)
                
                # 提取路径
                path = self._extract_path(goal_node_idx)
                return path, goal_cost
        
        # 如果达到最大迭代次数但未找到路径
        print(f"达到最大迭代次数 ({self.max_iterations})，返回到目标的最佳路径")
        print(f"最佳距离: {best_goal_distance:.6f}")
        
        # 如果我们至少有一个接近目标的节点
        if best_goal_distance < 0.1:  # 10cm的阈值
            # 尝试连接到实际目标
            if self._is_collision_free_cart(
                self.nodes_cart[best_goal_idx], 
                goal_ee_pos, 
                self.nodes_joint[best_goal_idx], 
                goal_joint_config
            ):
                print("连接到实际目标")
                goal_node_idx = len(self.nodes_cart)
                self.nodes_cart.append(goal_ee_pos)
                self.nodes_joint.append(goal_joint_config)
                goal_cost = self.costs[best_goal_idx] + self._distance_cart(self.nodes_cart[best_goal_idx], goal_ee_pos)
                self.costs.append(goal_cost)
                self.parents.append(best_goal_idx)
                
                # 提取路径
                path = self._extract_path(goal_node_idx)
                return path, goal_cost
        
        # 提取到最近节点的路径
        path = self._extract_path(best_goal_idx)
        return path, self.costs[best_goal_idx]
    
    def _extract_path(self, goal_idx: int) -> List[List[float]]:
        """从树中提取路径。
        
        Args:
            goal_idx: 目标节点索引
            
        Returns:
            关节位置列表，从起点到终点
        """
        path = []
        current_idx = goal_idx
        
        # 从目标到起点跟踪路径
        while current_idx != 0:
            path.append(self.nodes_joint[current_idx])
            current_idx = self.parents[current_idx]
            
        # 添加起点
        path.append(self.nodes_joint[0])
        
        # 反转路径以从起点到终点
        path.reverse()
        
        return path
    
    def clear_visualization(self) -> None:
        """清除所有可视化元素。"""
        for line_id in self.debug_lines:
            p.removeUserDebugItem(line_id)
        self.debug_lines = []
    
    def generate_smooth_trajectory(self, path: List[List[float]], smoothing_steps: int = 10) -> List[List[float]]:
        """生成平滑的轨迹。
        
        Args:
            path: 原始路径（关节位置列表）
            smoothing_steps: 在路径点之间插入的步数
            
        Returns:
            平滑的轨迹
        """
        if not path or len(path) < 2:
            return path
            
        smooth_path = []
        
        for i in range(len(path) - 1):
            start_config = path[i]
            end_config = path[i+1]
            
            for step in range(smoothing_steps):
                t = step / smoothing_steps
                # 线性插值
                config = [start + t * (end - start) for start, end in zip(start_config, end_config)]
                smooth_path.append(config)
                
        # 添加最后一个配置
        smooth_path.append(path[-1])
        
        return smooth_path


# 测试函数
def test_rrt_star_cartesian():
    """测试笛卡尔空间RRT*规划器"""
    import pybullet as p
    import pybullet_data
    import time

    # 初始化PyBullet
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    
    # 加载机器人和环境
    plane_id = p.loadURDF("plane.urdf")
    robot_id = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)
    
    # 设置机器人初始状态
    for i in range(p.getNumJoints(robot_id)):
        p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, 0)
    
    # 获取关节信息
    joint_indices = []
    lower_limits = []
    upper_limits = []
    
    for i in range(p.getNumJoints(robot_id)):
        joint_info = p.getJointInfo(robot_id, i)
        if joint_info[2] == p.JOINT_REVOLUTE:
            joint_indices.append(i)
            lower_limits.append(joint_info[8])
            upper_limits.append(joint_info[9])
    
    # 末端执行器索引
    ee_link_index = 11  # 假设末端执行器链接索引为11，根据实际机器人模型调整
    
    # 创建简单的障碍物跟踪器（模拟）
    class SimpleObstacleTracker:
        def __init__(self):
            self.obstacles = []
            # 添加一些障碍物
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
    
    # 创建规划器
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
    
    # 定义起始关节配置
    start_config = [0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0]
    
    # 定义目标位置和方向
    goal_pos = np.array([0.6, 0.2, 0.5])
    goal_orn = p.getQuaternionFromEuler([0, -np.pi/2, 0])
    
    # 可视化目标位置
    p.addUserDebugPoints([goal_pos], [[1, 0, 0]], pointSize=10)
    
    # 规划路径
    print("开始规划路径...")
    path, cost = planner.plan(start_config, goal_pos, goal_orn)
    
    if path:
        print(f"找到路径! 代价: {cost:.6f}")
        
        # 生成平滑轨迹
        smooth_path = planner.generate_smooth_trajectory(path, smoothing_steps=20)
        
        # 执行轨迹
        print("执行轨迹...")
        for joint_pos in smooth_path:
            # 设置关节位置
            for i, idx in enumerate(joint_indices):
                p.setJointMotorControl2(robot_id, idx, p.POSITION_CONTROL, joint_pos[i])
            
            # 更新仿真
            p.stepSimulation()
            time.sleep(0.01)
    else:
        print("未找到路径")
    
    # 保持窗口打开
    while p.isConnected():
        p.stepSimulation()
        time.sleep(0.01)
    

if __name__ == "__main__":
    test_rrt_star_cartesian() 