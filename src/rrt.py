import numpy as np
import pybullet as p
import time
import random
from queue import PriorityQueue
from math import sqrt


class Node:
    """
    RRT树节点类
    """
    def __init__(self, joint_angles, parent=None, cost=0.0):
        self.joint_angles = np.array(joint_angles)
        self.parent = parent
        self.cost = cost  # 从起点到当前节点的代价
        self.children = []  # 子节点列表


class RRTStarPlanner:
    """
    RRT*规划器
    """
    def __init__(self, robot_id, joint_indices, obstacle_ids=None, collision_fn=None):
        self.robot_id = robot_id
        self.joint_indices = joint_indices
        self.obstacle_ids = obstacle_ids if obstacle_ids else []
        self.collision_fn = collision_fn  # 自定义碰撞检测函数
        self.nodes = []
        self.goal_node = None
        self.solution_path = []
        self.smoothed_path = []
        self.goal_reached = False
        self.max_iterations = 1000
        self.step_size = 0.05  # 步长，控制扩展距离
        self.rewire_radius = 0.5  # 重新连接半径
        self.goal_sample_rate = 0.1  # 目标采样率
        self.final_path_smoothing = True  # 是否对最终路径进行平滑
        self.max_smoothing_attempts = 50  # 最大平滑尝试次数
        self.limit_ranges = None  # 机器人关节角度范围
        
    def set_joint_limits(self, limits):
        """
        设置关节限位
        limits: [(min_0, max_0), (min_1, max_1), ...]
        """
        self.limit_ranges = limits
        
    def set_collision_objects(self, obstacle_ids):
        """
        设置碰撞检测物体ID列表
        """
        self.obstacle_ids = obstacle_ids
        
    def set_step_size(self, step_size):
        """
        设置步长
        """
        self.step_size = step_size
        
    def set_rewire_radius(self, rewire_radius):
        """
        设置重新连接半径
        """
        self.rewire_radius = rewire_radius
        
    def set_goal_sample_rate(self, goal_sample_rate):
        """
        设置目标采样率
        """
        self.goal_sample_rate = goal_sample_rate
        
    def distance(self, from_node, to_node):
        """
        计算两个节点之间的欧几里得距离
        """
        return np.linalg.norm(np.array(from_node.joint_angles) - np.array(to_node.joint_angles))
    
    def steer(self, from_node, to_node):
        """
        在from_node到to_node方向上按步长扩展一个新节点
        """
        dist = self.distance(from_node, to_node)
        if dist < self.step_size:
            return to_node.joint_angles
        else:
            # 计算方向向量并按步长扩展
            direction = (to_node.joint_angles - from_node.joint_angles) / dist
            new_joint_angles = from_node.joint_angles + direction * self.step_size
            
            # 确保关节角度在限位范围内
            if self.limit_ranges:
                for i in range(len(new_joint_angles)):
                    min_val, max_val = self.limit_ranges[i]
                    new_joint_angles[i] = max(min_val, min(max_val, new_joint_angles[i]))
                    
            return new_joint_angles
            
    def check_collision(self, joint_angles):
        """
        检查给定关节角度是否会导致碰撞
        """
        # 保存当前机器人状态
        original_joints = []
        for idx in self.joint_indices:
            original_joints.append(p.getJointState(self.robot_id, idx)[0])
        
        # 设置关节角度以进行碰撞检测
        for i, joint_idx in enumerate(self.joint_indices):
            p.resetJointState(self.robot_id, joint_idx, joint_angles[i])
            
        # 进行碰撞检测
        collision = False
        
        # 如果提供了自定义碰撞检测函数，使用它
        if self.collision_fn:
            collision = self.collision_fn()
        else:
            # 默认的碰撞检测逻辑
            for obstacle_id in self.obstacle_ids:
                if p.getContactPoints(self.robot_id, obstacle_id):
                    collision = True
                    break
        
        # 恢复机器人状态
        for i, joint_idx in enumerate(self.joint_indices):
            p.resetJointState(self.robot_id, joint_idx, original_joints[i])
            
        return collision
    
    def check_path_collision(self, from_node, to_node, steps=10):
        """
        检查从from_node到to_node的路径是否会有碰撞
        """
        from_angles = from_node.joint_angles
        to_angles = to_node.joint_angles
        
        for i in range(1, steps+1):
            t = i / steps
            # 线性插值
            interpolated_angles = from_angles + t * (to_angles - from_angles)
            if self.check_collision(interpolated_angles):
                return True  # 有碰撞
                
        return False  # 无碰撞
        
    def sample_random_node(self, goal_node):
        """
        采样随机节点，有一定概率直接返回目标节点
        """
        if random.random() < self.goal_sample_rate:
            return Node(goal_node.joint_angles)
        else:
            # 随机采样关节角度
            if self.limit_ranges:
                random_angles = []
                for min_val, max_val in self.limit_ranges:
                    random_angles.append(random.uniform(min_val, max_val))
                return Node(random_angles)
            else:
                # 如果未设置限位，使用当前最大节点范围的2倍作为采样范围
                if not self.nodes:
                    raise ValueError("Cannot sample without joint limits or existing nodes")
                
                all_angles = np.array([node.joint_angles for node in self.nodes])
                min_vals = np.min(all_angles, axis=0)
                max_vals = np.max(all_angles, axis=0)
                range_vals = max_vals - min_vals
                
                # 确保有一定的采样范围
                min_range = 0.1
                range_vals = np.maximum(range_vals, min_range)
                
                center = (min_vals + max_vals) / 2
                random_angles = []
                for i in range(len(center)):
                    random_angles.append(random.uniform(center[i] - range_vals[i], 
                                                       center[i] + range_vals[i]))
                return Node(random_angles)
    
    def find_nearest_node(self, point_node):
        """
        找到与给定节点最近的树中节点
        """
        min_dist = float('inf')
        nearest_node = None
        
        for node in self.nodes:
            dist = self.distance(node, point_node)
            if dist < min_dist:
                min_dist = dist
                nearest_node = node
                
        return nearest_node
    
    def find_near_nodes(self, new_node):
        """
        找到与新节点在rewire_radius范围内的所有节点
        """
        near_nodes = []
        
        for node in self.nodes:
            if self.distance(node, new_node) <= self.rewire_radius:
                near_nodes.append(node)
                
        return near_nodes
    
    def calculate_cost(self, from_node, to_node):
        """
        计算从from_node到to_node的代价（这里使用欧几里得距离）
        """
        return self.distance(from_node, to_node)
    
    def choose_parent(self, new_node, near_nodes):
        """
        从近邻节点中选择最优父节点
        """
        if not near_nodes:
            return None
            
        costs = []
        for near_node in near_nodes:
            # 检查从near_node到new_node的路径是否无碰撞
            if not self.check_path_collision(near_node, new_node):
                # 计算经过near_node到达new_node的总代价
                cost = near_node.cost + self.calculate_cost(near_node, new_node)
                costs.append((near_node, cost))
        
        if not costs:
            return None
            
        # 选择总代价最小的节点作为父节点
        best_node, best_cost = min(costs, key=lambda x: x[1])
        new_node.parent = best_node
        new_node.cost = best_cost
        return best_node
    
    def rewire(self, new_node, near_nodes):
        """
        重新连接 - 检查是否可以通过新节点改善近邻节点的路径
        """
        for near_node in near_nodes:
            # 如果near_node是新节点的父节点或祖先节点，跳过
            temp_node = new_node.parent
            is_ancestor = False
            while temp_node:
                if temp_node == near_node:
                    is_ancestor = True
                    break
                temp_node = temp_node.parent
                
            if is_ancestor:
                continue
                
            # 计算通过新节点到达近邻节点的代价
            potential_cost = new_node.cost + self.calculate_cost(new_node, near_node)
            
            # 如果新路径更优且无碰撞，则重新连接
            if potential_cost < near_node.cost and not self.check_path_collision(new_node, near_node):
                if near_node.parent:
                    near_node.parent.children.remove(near_node)
                near_node.parent = new_node
                near_node.cost = potential_cost
                new_node.children.append(near_node)
    
    def extract_path(self, goal_node):
        """
        提取从起点到目标节点的路径
        """
        path = []
        current = goal_node
        
        while current:
            path.append(current.joint_angles)
            current = current.parent
            
        return path[::-1]  # 反转路径，使其从起点到终点
    
    def smooth_path(self, path, attempts=None):
        """
        对路径进行平滑处理
        """
        if attempts is None:
            attempts = self.max_smoothing_attempts
            
        if len(path) <= 2:
            return path
            
        smoothed = [path[0]]
        i = 0
        
        while i < len(path) - 1:
            current = path[i]
            
            # 尝试跳过中间点，直接连接到更远的点
            for j in range(len(path) - 1, i, -1):
                if j - i <= 1:  # 相邻点无需检查
                    continue
                    
                # 检查从current到path[j]的路径是否无碰撞
                if not self.check_path_collision(Node(current), Node(path[j])):
                    # 可以跳过中间点
                    i = j - 1
                    break
                    
            i += 1
            if i < len(path):
                smoothed.append(path[i])
                
        return smoothed
    
    def plan(self, start_joint_angles, goal_joint_angles, max_iterations=None, timeout=60):
        """
        执行RRT*规划算法
        
        Parameters:
        -----------
        start_joint_angles: 起始关节角度
        goal_joint_angles: 目标关节角度
        max_iterations: 最大迭代次数
        timeout: 超时时间(秒)
        
        Returns:
        --------
        path: 关节空间轨迹点列表，如果规划失败则为空列表
        """
        if max_iterations is not None:
            self.max_iterations = max_iterations
            
        # 初始化起点和目标
        start_node = Node(start_joint_angles, cost=0.0)
        goal_node = Node(goal_joint_angles)
        
        # 重置规划器状态
        self.nodes = [start_node]
        self.goal_node = None
        self.solution_path = []
        self.smoothed_path = []
        self.goal_reached = False
        
        start_time = time.time()
        
        # 主循环
        for i in range(self.max_iterations):
            # 检查超时
            if time.time() - start_time > timeout:
                print(f"RRT* planning timed out after {timeout} seconds")
                break
                
            # 随机采样
            random_node = self.sample_random_node(goal_node)
            
            # 找到最近节点
            nearest_node = self.find_nearest_node(random_node)
            
            if nearest_node is None:
                continue
                
            # 按步长扩展
            new_joint_angles = self.steer(nearest_node, random_node)
            new_node = Node(new_joint_angles)
            
            # 碰撞检测
            if self.check_collision(new_joint_angles):
                continue
                
            # 路径碰撞检测
            if self.check_path_collision(nearest_node, new_node):
                continue
                
            # 寻找近邻节点
            near_nodes = self.find_near_nodes(new_node)
            
            # 选择最优父节点
            best_parent = self.choose_parent(new_node, near_nodes)
            
            if best_parent is None:
                # 如果找不到有效父节点，使用最近节点作为父节点
                if not self.check_path_collision(nearest_node, new_node):
                    new_node.parent = nearest_node
                    new_node.cost = nearest_node.cost + self.calculate_cost(nearest_node, new_node)
                    nearest_node.children.append(new_node)
                    self.nodes.append(new_node)
                else:
                    continue
            else:
                best_parent.children.append(new_node)
                self.nodes.append(new_node)
                
                # 执行重新连接
                self.rewire(new_node, near_nodes)
            
            # 检查是否可以连接到目标
            dist_to_goal = self.distance(new_node, goal_node)
            if dist_to_goal <= self.step_size:
                goal_copy = Node(goal_node.joint_angles)
                goal_copy.parent = new_node
                goal_copy.cost = new_node.cost + dist_to_goal
                
                if not self.check_path_collision(new_node, goal_copy):
                    self.goal_node = goal_copy
                    self.goal_reached = True
                    print(f"Goal reached after {i+1} iterations!")
                    break
        
        # 提取路径
        if self.goal_reached:
            self.solution_path = self.extract_path(self.goal_node)
            
            # 路径平滑
            if self.final_path_smoothing:
                self.smoothed_path = self.smooth_path(self.solution_path)
                return self.smoothed_path
            else:
                return self.solution_path
        else:
            print(f"Failed to reach goal after {self.max_iterations} iterations")
            
            # 尝试找到最接近目标的节点
            min_dist = float('inf')
            closest_node = None
            for node in self.nodes:
                dist = self.distance(node, goal_node)
                if dist < min_dist:
                    min_dist = dist
                    closest_node = node
            
            if closest_node:
                print(f"Returning path to closest node (distance to goal: {min_dist})")
                self.solution_path = self.extract_path(closest_node)
                
                if self.final_path_smoothing:
                    self.smoothed_path = self.smooth_path(self.solution_path)
                    return self.smoothed_path
                else:
                    return self.solution_path
            
            return []


class RRTConnectPlanner:
    """
    RRT-Connect规划器 - 双向快速RRT变体
    """
    def __init__(self, robot_id, joint_indices, obstacle_ids=None, collision_fn=None):
        self.robot_id = robot_id
        self.joint_indices = joint_indices
        self.obstacle_ids = obstacle_ids if obstacle_ids else []
        self.collision_fn = collision_fn
        self.trees = [[], []]  # 起点树和终点树
        self.max_iterations = 1000
        self.step_size = 0.05
        self.final_path_smoothing = True
        self.max_smoothing_attempts = 50
        self.limit_ranges = None
        
    def set_joint_limits(self, limits):
        """设置关节限位"""
        self.limit_ranges = limits
        
    def set_collision_objects(self, obstacle_ids):
        """设置碰撞检测物体ID列表"""
        self.obstacle_ids = obstacle_ids
        
    def set_step_size(self, step_size):
        """设置步长"""
        self.step_size = step_size
        
    def distance(self, from_node, to_node):
        """计算两个节点之间的欧几里得距离"""
        return np.linalg.norm(np.array(from_node.joint_angles) - np.array(to_node.joint_angles))
    
    def steer(self, from_node, to_node):
        """
        在from_node到to_node方向上按步长扩展一个新节点
        """
        dist = self.distance(from_node, to_node)
        if dist < self.step_size:
            return to_node.joint_angles
        else:
            # 计算方向向量并按步长扩展
            direction = (to_node.joint_angles - from_node.joint_angles) / dist
            new_joint_angles = from_node.joint_angles + direction * self.step_size
            
            # 确保关节角度在限位范围内
            if self.limit_ranges:
                for i in range(len(new_joint_angles)):
                    min_val, max_val = self.limit_ranges[i]
                    new_joint_angles[i] = max(min_val, min(max_val, new_joint_angles[i]))
                    
            return new_joint_angles
            
    def check_collision(self, joint_angles):
        """
        检查给定关节角度是否会导致碰撞
        """
        # 保存当前机器人状态
        original_joints = []
        for idx in self.joint_indices:
            original_joints.append(p.getJointState(self.robot_id, idx)[0])
        
        # 设置关节角度以进行碰撞检测
        for i, joint_idx in enumerate(self.joint_indices):
            p.resetJointState(self.robot_id, joint_idx, joint_angles[i])
            
        # 进行碰撞检测
        collision = False
        
        # 如果提供了自定义碰撞检测函数，使用它
        if self.collision_fn:
            collision = self.collision_fn()
        else:
            # 默认的碰撞检测逻辑
            for obstacle_id in self.obstacle_ids:
                if p.getContactPoints(self.robot_id, obstacle_id):
                    collision = True
                    break
        
        # 恢复机器人状态
        for i, joint_idx in enumerate(self.joint_indices):
            p.resetJointState(self.robot_id, joint_idx, original_joints[i])
            
        return collision
    
    def check_path_collision(self, from_node, to_node, steps=10):
        """
        检查从from_node到to_node的路径是否会有碰撞
        """
        from_angles = from_node.joint_angles
        to_angles = to_node.joint_angles
        
        for i in range(1, steps+1):
            t = i / steps
            # 线性插值
            interpolated_angles = from_angles + t * (to_angles - from_angles)
            if self.check_collision(interpolated_angles):
                return True  # 有碰撞
                
        return False  # 无碰撞
    
    def sample_random_node(self):
        """采样随机节点"""
        if self.limit_ranges:
            random_angles = []
            for min_val, max_val in self.limit_ranges:
                random_angles.append(random.uniform(min_val, max_val))
            return Node(random_angles)
        else:
            # 如果未设置限位，使用当前节点范围的2倍作为采样范围
            all_angles = []
            for tree in self.trees:
                all_angles.extend([node.joint_angles for node in tree])
                
            if not all_angles:
                raise ValueError("Cannot sample without joint limits or existing nodes")
                
            all_angles = np.array(all_angles)
            min_vals = np.min(all_angles, axis=0)
            max_vals = np.max(all_angles, axis=0)
            range_vals = max_vals - min_vals
            
            # 确保有一定的采样范围
            min_range = 0.1
            range_vals = np.maximum(range_vals, min_range)
            
            center = (min_vals + max_vals) / 2
            random_angles = []
            for i in range(len(center)):
                random_angles.append(random.uniform(center[i] - range_vals[i], 
                                                   center[i] + range_vals[i]))
            return Node(random_angles)
    
    def find_nearest_node(self, point_node, tree_idx):
        """找到与给定节点最近的树中节点"""
        min_dist = float('inf')
        nearest_node = None
        
        for node in self.trees[tree_idx]:
            dist = self.distance(node, point_node)
            if dist < min_dist:
                min_dist = dist
                nearest_node = node
                
        return nearest_node
    
    def extend(self, tree_idx, target_node):
        """
        向目标节点方向扩展一步
        返回状态：
        - 'advanced': 向目标方向前进了一步
        - 'reached': 到达了目标
        - 'trapped': 被障碍物阻挡
        """
        nearest_node = self.find_nearest_node(target_node, tree_idx)
        
        if nearest_node is None:
            return 'trapped', None
            
        # 按步长扩展
        new_joint_angles = self.steer(nearest_node, target_node)
        new_node = Node(new_joint_angles, parent=nearest_node)
        
        # 碰撞检测
        if self.check_collision(new_joint_angles):
            return 'trapped', None
            
        # 路径碰撞检测
        if self.check_path_collision(nearest_node, new_node):
            return 'trapped', None
            
        # 添加到树中
        self.trees[tree_idx].append(new_node)
        
        # 检查是否达到目标
        if self.distance(new_node, target_node) < self.step_size:
            return 'reached', new_node
        else:
            return 'advanced', new_node
    
    def connect(self, tree_idx, target_node):
        """
        尝试连接到目标节点，持续使用extend直到'trapped'或'reached'
        """
        result = self.extend(tree_idx, target_node)
        status, new_node = result
        
        while status == 'advanced':
            result = self.extend(tree_idx, target_node)
            status, new_node = result
            
        return status, new_node
    
    def extract_path(self, node_a, node_b=None):
        """
        提取路径
        如果node_b为None，则提取从根到node_a的路径
        否则，提取从node_a到node_b的双向路径
        """
        if node_b is None:
            # 单向路径
            path = []
            current = node_a
            
            while current:
                path.append(current.joint_angles)
                current = current.parent
                
            return path[::-1]  # 反转路径，使其从起点到终点
        else:
            # 双向路径，node_a来自起点树，node_b来自终点树
            path_a = []
            current = node_a
            
            while current:
                path_a.append(current.joint_angles)
                current = current.parent
                
            path_a = path_a[::-1]  # 反转，使其从起点到连接点
            
            path_b = []
            current = node_b
            
            while current:
                path_b.append(current.joint_angles)
                current = current.parent
                
            # path_b已经是从连接点到终点的顺序
            
            # 组合路径
            return path_a + path_b
    
    def smooth_path(self, path, attempts=None):
        """
        对路径进行平滑处理
        """
        if attempts is None:
            attempts = self.max_smoothing_attempts
            
        if len(path) <= 2:
            return path
            
        smoothed = [path[0]]
        i = 0
        
        while i < len(path) - 1:
            current = path[i]
            
            # 尝试跳过中间点，直接连接到更远的点
            for j in range(len(path) - 1, i, -1):
                if j - i <= 1:  # 相邻点无需检查
                    continue
                    
                # 检查从current到path[j]的路径是否无碰撞
                if not self.check_path_collision(Node(current), Node(path[j])):
                    # 可以跳过中间点
                    i = j - 1
                    break
                    
            i += 1
            if i < len(path):
                smoothed.append(path[i])
                
        return smoothed
    
    def plan(self, start_joint_angles, goal_joint_angles, max_iterations=None, timeout=60):
        """
        执行RRT-Connect规划算法
        
        Parameters:
        -----------
        start_joint_angles: 起始关节角度
        goal_joint_angles: 目标关节角度
        max_iterations: 最大迭代次数
        timeout: 超时时间(秒)
        
        Returns:
        --------
        path: 关节空间轨迹点列表，如果规划失败则为空列表
        """
        if max_iterations is not None:
            self.max_iterations = max_iterations
            
        # 初始化起点和终点树
        start_node = Node(start_joint_angles)
        goal_node = Node(goal_joint_angles)
        self.trees[0] = [start_node]  # 起点树
        self.trees[1] = [goal_node]   # 终点树
        
        start_time = time.time()
        
        # 主循环
        for i in range(self.max_iterations):
            # 检查超时
            if time.time() - start_time > timeout:
                print(f"RRT-Connect planning timed out after {timeout} seconds")
                break
                
            # 随机采样
            random_node = self.sample_random_node()
            
            # 从起点树扩展到随机点
            status_a, new_node_a = self.extend(0, random_node)
            
            if status_a != 'trapped':
                # 尝试从终点树连接到新节点
                status_b, new_node_b = self.connect(1, new_node_a)
                
                if status_b == 'reached':
                    # 成功连接，提取路径
                    path = self.extract_path(new_node_a, new_node_b)
                    
                    # 路径平滑
                    if self.final_path_smoothing:
                        path = self.smooth_path(path)
                        
                    print(f"Path found after {i+1} iterations with {len(path)} waypoints!")
                    return path
            
            # 交换两棵树的角色
            self.trees.reverse()
        
        print(f"Failed to find path after {self.max_iterations} iterations")
        return []