import numpy as np
from typing import List, Tuple, Dict, Optional
import pybullet as p
import time

class TreeNode:
    """RRT树节点"""
    def __init__(self, config: np.ndarray):
        self.config = config  # 关节配置
        self.cost = 0.0      # 从起点到该节点的代价
        self.parent = None   # 父节点
        self.children = []   # 子节点列表

class RRTTree:
    """RRT树结构"""
    def __init__(self, root_config: np.ndarray):
        self.root = TreeNode(root_config)
        self.nodes = [self.root]
        
    def add_node(self, config: np.ndarray, parent: TreeNode) -> TreeNode:
        node = TreeNode(config)
        node.parent = parent
        parent.children.append(node)
        self.nodes.append(node)
        return node
        
    def remove_edge(self, parent: TreeNode, child: TreeNode):
        parent.children.remove(child)
        child.parent = None
        
    def add_edge(self, parent: TreeNode, child: TreeNode):
        parent.children.append(child)
        child.parent = parent

class RRTStarConnect:
    """RRT* Connect 规划器"""
    def __init__(self, 
                 robot,
                 obstacle_tracker,
                 goal_region: Dict,
                 max_iters: int = 1000,
                 step_size: float = 0.1,
                 neighbor_radius: float = 0.5,
                 goal_bias: float = 0.1):
        
        self.robot = robot
        self.obstacle_tracker = obstacle_tracker
        self.goal_region = goal_region
        
        # Planning parameters
        self.max_iters = max_iters
        self.step_size = step_size
        self.neighbor_radius = neighbor_radius
        self.goal_bias = goal_bias
        
    def config_cost(self, config1: np.ndarray, config2: np.ndarray) -> float:
        """计算两个配置之间的代价（这里使用关节空间距离）"""
        return np.linalg.norm(config2 - config1)

    def nearest_neighbor(self, tree: RRTTree, config: np.ndarray) -> TreeNode:
        """找到树中最近的节点"""
        distances = [self.config_cost(node.config, config) for node in tree.nodes]
        return tree.nodes[np.argmin(distances)]

    def steer(self, from_config: np.ndarray, to_config: np.ndarray) -> np.ndarray:
        """在两个配置之间进行插值"""
        diff = to_config - from_config
        dist = np.linalg.norm(diff)
        if dist > self.step_size:
            return from_config + (diff / dist) * self.step_size
        return to_config

    def check_collision(self, config: np.ndarray) -> bool:
        """检查配置是否碰撞"""
        # 临时设置机器人配置
        original_joints = self.robot.get_joint_positions()
        for idx, pos in zip(self.robot.arm_idx, config):
            p.resetJointState(self.robot.id, idx, pos)

        # 获取障碍物状态
        obstacles = self.obstacle_tracker.get_all_obstacle_states()
        
        in_collision = False
        for obstacle in obstacles:
            if obstacle is None:
                continue
                
            predicted_pos = obstacle['position']
            radius = obstacle['radius']
            
            # 检查与障碍物的碰撞
            closest_points = p.getClosestPoints(
                self.robot.id,
                -1,  # 使用ray-test检查与球体障碍物的碰撞
                distance=radius * 1.1,  # 略大于障碍物半径
                rayFromPosition=predicted_pos,
                rayToPosition=predicted_pos + np.array([0, 0, 0.001])
            )
            
            if closest_points and any(pt[8] < radius for pt in closest_points):
                in_collision = True
                break

        # 恢复原始配置
        for idx, pos in zip(self.robot.arm_idx, original_joints):
            p.resetJointState(self.robot.id, idx, pos)

        return in_collision

    def near_nodes(self, tree: RRTTree, config: np.ndarray) -> List[TreeNode]:
        """返回邻域内的所有节点"""
        return [node for node in tree.nodes 
                if self.config_cost(node.config, config) < self.neighbor_radius]

    def extend_rrt_star(self, tree: RRTTree, target_config: np.ndarray) -> Optional[TreeNode]:
        """RRT*扩展"""
        # 1. 找最近节点
        nearest = self.nearest_neighbor(tree, target_config)
        
        # 2. 朝目标方向扩展
        new_config = self.steer(nearest.config, target_config)
        
        # 3. 检查碰撞
        if self.check_collision(new_config):
            return None
            
        # 4. 在半径范围内找近邻节点
        near_nodes = self.near_nodes(tree, new_config)
        
        # 5. 选择最优父节点
        min_cost = float('inf')
        min_parent = None
        
        for near_node in near_nodes:
            potential_cost = (near_node.cost + 
                            self.config_cost(near_node.config, new_config))
            if potential_cost < min_cost and not self.check_collision(new_config):
                min_cost = potential_cost
                min_parent = near_node
        
        if min_parent is None:
            min_parent = nearest
            min_cost = nearest.cost + self.config_cost(nearest.config, new_config)
        
        # 6. 添加新节点
        new_node = tree.add_node(new_config, min_parent)
        new_node.cost = min_cost
        
        # 7. 重新布线
        self.rewire(tree, new_node, near_nodes)
        
        return new_node

    def rewire(self, tree: RRTTree, new_node: TreeNode, near_nodes: List[TreeNode]):
        """对邻近节点进行重新布线"""
        for near_node in near_nodes:
            if near_node == new_node.parent:
                continue
                
            potential_cost = (new_node.cost + 
                            self.config_cost(new_node.config, near_node.config))
                            
            if potential_cost < near_node.cost and not self.check_collision(near_node.config):
                old_parent = near_node.parent
                # 更新边
                tree.remove_edge(old_parent, near_node)
                tree.add_edge(new_node, near_node)
                # 更新代价
                cost_diff = potential_cost - near_node.cost
                self.update_descendants_cost(near_node, cost_diff)

    def update_descendants_cost(self, node: TreeNode, cost_diff: float):
        """更新子树中所有节点的代价"""
        for child in node.children:
            child.cost += cost_diff
            self.update_descendants_cost(child, cost_diff)

    def connect(self, tree: RRTTree, target_config: np.ndarray) -> Tuple[str, Optional[TreeNode]]:
        """尝试连接到目标配置"""
        node = self.extend_rrt_star(tree, target_config)
        if node is None:
            return "Trapped", None
            
        while True:
            dist = self.config_cost(node.config, target_config)
            if dist < self.step_size:
                return "Reached", node
                
            new_node = self.extend_rrt_star(tree, target_config)
            if new_node is None:
                return "Trapped", node
            node = new_node

    def plan(self, start_config: np.ndarray, goal_config: np.ndarray) -> Optional[List[np.ndarray]]:
        """主规划函数"""
        print("\nStarting RRT* Connect planning...")
        # 初始化两棵树
        start_tree = RRTTree(start_config)
        goal_tree = RRTTree(goal_config)
        
        for i in range(self.max_iters):
            # 采样新配置
            if np.random.random() < self.goal_bias:
                rand_config = goal_config
            else:
                rand_config = self.sample_config()
            
            # 扩展起始树
            new_node = self.extend_rrt_star(start_tree, rand_config)
            if new_node is not None:
                # 尝试连接目标树
                status, node = self.connect(goal_tree, new_node.config)
                if status == "Reached":
                    return self.extract_path(start_tree, goal_tree, new_node, node)
            
            # 交换两棵树
            start_tree, goal_tree = goal_tree, start_tree
            
            if i % 100 == 0:  # 每100次迭代打印一次进度
                print(f"Iteration {i}, Tree sizes: {len(start_tree.nodes)}, {len(goal_tree.nodes)}")
        return None

    def sample_config(self) -> np.ndarray:
        """采样一个随机配置"""
        config = np.zeros(len(self.robot.arm_idx))
        for i in range(len(config)):
            config[i] = np.random.uniform(
                self.robot.lower_limits[i],
                self.robot.upper_limits[i]
            )
        return config

    def extract_path(self, 
                    tree_a: RRTTree, 
                    tree_b: RRTTree, 
                    node_a: TreeNode, 
                    node_b: TreeNode) -> List[np.ndarray]:
        """提取路径"""
        path_a = []
        current = node_a
        while current is not None:
            path_a.append(current.config)
            current = current.parent
        path_a = path_a[::-1]
        
        path_b = []
        current = node_b
        while current is not None:
            path_b.append(current.config)
            current = current.parent
            
        return path_a + path_b

