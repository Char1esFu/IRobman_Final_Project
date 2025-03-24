import numpy as np
import pybullet as p
import random
import time
from typing import List, Tuple, Dict, Optional, Any, Callable

class PotentialFieldPlanner:
    """
    Potential Field Planner for robotic arm.
    
    Plans in joint space while performing collision detection in Cartesian space.
    Can be used for dynamic obstacle avoidance in combination with global RRT* path.
    
    Args:
        robot_id: PyBullet robot ID
        joint_indices: List of joint indices to control
        lower_limits: Lower joint limits
        upper_limits: Upper joint limits
        ee_link_index: End effector link index
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
        robot_id: int,
        joint_indices: List[int],
        lower_limits: List[float],
        upper_limits: List[float],
        ee_link_index: int,
        obstacle_tracker: Any,
        max_iterations: int = 300,
        step_size: float = 0.01,
        d0: float = 0.2,
        K_att: float = 1.0,
        K_rep: float = 1.0,
        goal_threshold: float = 0.05,
        collision_check_step: float = 0.05,
        reference_path_weight: float = 0.5
    ):
        self.robot_id = robot_id
        self.joint_indices = joint_indices
        self.lower_limits = lower_limits
        self.upper_limits = upper_limits
        self.ee_link_index = ee_link_index
        self.obstacle_tracker = obstacle_tracker
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.d0 = d0  # 障碍物影响距离
        self.K_att = K_att  # 吸引势场增益
        self.K_rep = K_rep  # 排斥势场增益
        self.goal_threshold = goal_threshold
        self.collision_check_step = collision_check_step
        self.reference_path_weight = reference_path_weight
        
        self.dimension = len(joint_indices)
        
        # 可视化线条
        self.debug_lines = []
        
        # 存储可能的全局参考轨迹
        self.reference_path = None
    
    def _get_current_ee_pose(self, joint_positions: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """获取指定关节位置的末端执行器姿态
        
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
    
    def _is_collision_free(self, joint_pos: List[float]) -> bool:
        """检查关节位置是否无碰撞
        
        Args:
            joint_pos: 要检查的关节位置
            
        Returns:
            如果无碰撞，则为True，否则为False
        """
        # 获取末端执行器位置和方向
        ee_pos, _ = self._get_current_ee_pose(joint_pos)
        
        # 获取要检查的机器人链接位置
        links_to_check = self.joint_indices + [self.ee_link_index]
        
        # 保存当前状态
        current_states = []
        for i in self.joint_indices:
            current_states.append(p.getJointState(self.robot_id, i)[0])
            
        # 设置关节位置
        for i, idx in enumerate(self.joint_indices):
            p.resetJointState(self.robot_id, idx, joint_pos[i])
        
        # 检查障碍物碰撞
        collision = False
        
        # 从跟踪器获取障碍物状态
        obstacle_states = self.obstacle_tracker.get_all_obstacle_states()
        if obstacle_states is None or len(obstacle_states) == 0:
            # 恢复原始状态
            for i, idx in enumerate(self.joint_indices):
                p.resetJointState(self.robot_id, idx, current_states[i])
            return True  # 无障碍物，无碰撞
        
        # 检查每个链接与每个障碍物的碰撞
        for link_idx in links_to_check:
            link_state = p.getLinkState(self.robot_id, link_idx)
            link_pos = np.array(link_state[0])
            
            for obstacle in obstacle_states:
                if obstacle is None:
                    continue
                
                # 简单的球体碰撞检查
                obstacle_pos = obstacle['position']
                obstacle_radius = obstacle['radius']
                
                # 链接与障碍物中心的距离
                dist = np.linalg.norm(link_pos - obstacle_pos)
                
                # 将机器人链接近似为点（简化）
                # 添加小的安全边距（0.05m）
                if dist < obstacle_radius + 0.05:
                    collision = True
                    break
            
            if collision:
                break
                
        # 恢复原始状态
        for i, idx in enumerate(self.joint_indices):
            p.resetJointState(self.robot_id, idx, current_states[i])
            
        return not collision
    
    def _distance(self, q1: List[float], q2: List[float]) -> float:
        """计算两个关节配置之间的距离
        
        Args:
            q1: 第一个关节配置
            q2: 第二个关节配置
            
        Returns:
            关节空间中的欧几里得距离
        """
        return np.linalg.norm(np.array(q1) - np.array(q2))
    

    
    def _attractive_gradient(self, q: np.ndarray, q_goal: np.ndarray) -> np.ndarray:
        """计算吸引势能梯度
        
        Args:
            q: 当前关节配置
            q_goal: 目标关节配置
            
        Returns:
            吸引势能梯度（指向目标）
        """
        return self.K_att * (q - q_goal)
    
    def _repulsive_potential(self, q: np.ndarray) -> float:
        """计算排斥势能
        
        Args:
            q: 当前关节配置
            
        Returns:
            排斥势能值
        """
        # 保存当前状态
        current_states = []
        for i in self.joint_indices:
            current_states.append(p.getJointState(self.robot_id, i)[0])
            
        # 设置关节位置
        for i, idx in enumerate(self.joint_indices):
            p.resetJointState(self.robot_id, idx, q[i])
        
        # 要检查的链接
        links_to_check = self.joint_indices + [self.ee_link_index]
        
        # 从跟踪器获取障碍物状态
        obstacle_states = self.obstacle_tracker.get_all_obstacle_states()
        
        potential = 0.0
        
        if obstacle_states:
            for link_idx in links_to_check:
                link_state = p.getLinkState(self.robot_id, link_idx)
                link_pos = np.array(link_state[0])
                
                for obstacle in obstacle_states:
                    if obstacle is None:
                        continue
                    
                    obstacle_pos = obstacle['position']
                    obstacle_radius = obstacle['radius']
                    
                    # 链接与障碍物中心的距离
                    dist = np.linalg.norm(link_pos - obstacle_pos) - obstacle_radius
                    
                    # 如果在影响范围内，计算排斥势
                    if dist < self.d0:
                        if dist < 0.01:  # 防止除以非常小的数
                            dist = 0.01
                        
                        # 排斥势公式: 0.5 * K_rep * (1/dist - 1/d0)^2
                        potential += 0.5 * self.K_rep * ((1.0 / dist) - (1.0 / self.d0))**2
        
        # 恢复原始状态
        for i, idx in enumerate(self.joint_indices):
            p.resetJointState(self.robot_id, idx, current_states[i])
            
        return potential
    
    def _repulsive_gradient(self, q: np.ndarray) -> np.ndarray:
        """计算排斥势能梯度
        
        Args:
            q: 当前关节配置
            
        Returns:
            排斥势能梯度（远离障碍物）
        """
        # 数值微分计算梯度
        grad = np.zeros(self.dimension)
        epsilon = 1e-3  # 小的扰动
        
        for i in range(self.dimension):
            q_plus = q.copy()
            q_plus[i] += epsilon
            
            q_minus = q.copy()
            q_minus[i] -= epsilon
            
            # 中心差分
            grad[i] = (self._repulsive_potential(q_plus) - self._repulsive_potential(q_minus)) / (2 * epsilon)
        
        return grad
    
    def _reference_path_gradient(self, q: np.ndarray) -> np.ndarray:
        """计算参考路径的梯度（引导机器人跟随RRT*全局路径）
        
        Args:
            q: 当前关节配置
            
        Returns:
            参考路径梯度
        """
        if self.reference_path is None or len(self.reference_path) < 2:
            return np.zeros(self.dimension)
        
        # 找到参考路径上最近的点
        distances = [self._distance(q, np.array(p)) for p in self.reference_path]
        min_idx = np.argmin(distances)
        
        # 如果已经是最后一点，指向最后一点
        if min_idx == len(self.reference_path) - 1:
            closest_point = np.array(self.reference_path[min_idx])
            return self.reference_path_weight * self.K_att * (q - closest_point)
        
        # 否则，指向下一个路径点
        next_point = np.array(self.reference_path[min_idx + 1])
        return self.reference_path_weight * self.K_att * (q - next_point)
    
    def _total_gradient(self, q: np.ndarray, q_goal: np.ndarray ,reference) -> np.ndarray:
        """计算总梯度（吸引 + 排斥 + 参考路径）
        
        Args:
            q: 当前关节配置
            q_goal: 目标关节配置
            
        Returns:
            总梯度
        """
        # 吸引梯度（指向目标）
        att_grad = -self._attractive_gradient(q, q_goal)
        
        # 排斥梯度（远离障碍物）
        rep_grad = -self._repulsive_gradient(q)
        
        if reference :
            # 参考路径梯度（引导跟随RRT*路径）
            ref_grad = -self._reference_path_gradient(q)
            # 总梯度 = 吸引 + 排斥 + 参考路径
            total_grad = att_grad + rep_grad + ref_grad
        else:
            total_grad = att_grad + rep_grad
        
        # 归一化梯度
        norm = np.linalg.norm(total_grad)
        if norm > 1e-6:  # 防止除以零
            total_grad = total_grad / norm
        
        return total_grad
    
    
    def plan_next_step(self, current_config: List[float], goal_config: List[float], reference: bool) -> Tuple[List[float], float]:
        """计算从当前位置出发的下一个最佳步骤
        
        这个方法更适合动态环境，只关注当前的局部最优方向
        
        Args:
            current_config: 当前关节配置
            goal_config: 目标关节配置
            reference: 是否有参考轨迹
            
        Returns:
            元组 (下一步关节配置, 到目标的距离)
        """
        q_current = np.array(current_config)
        q_goal = np.array(goal_config)
        
        # 计算梯度方向
        gradient = self._total_gradient(q_current, q_goal,reference)
        
        # 根据梯度更新位置
        q_new = q_current + self.step_size * gradient
        
        # 强制保持在关节限制内
        for j in range(self.dimension):
            q_new[j] = max(self.lower_limits[j], min(self.upper_limits[j], q_new[j]))
        
        # 检查是否无碰撞，如果有碰撞则尝试只使用排斥力
        if not self._is_collision_free(q_new):
            rep_grad = -self._repulsive_gradient(q_current)
            norm_rep = np.linalg.norm(rep_grad)
            if norm_rep > 1e-6:
                rep_grad = rep_grad / norm_rep
                q_new = q_current + self.step_size * rep_grad
                
                # 强制保持在关节限制内
                for j in range(self.dimension):
                    q_new[j] = max(self.lower_limits[j], min(self.upper_limits[j], q_new[j]))
                
                # 如果仍然有碰撞，返回当前位置
                if not self._is_collision_free(q_new):
                    q_new = q_current
        
        # 计算到目标的距离
        ee_pos, _ = self._get_current_ee_pose(q_new.tolist())
        goal_ee_pos, _ = self._get_current_ee_pose(goal_config)
        cost = np.linalg.norm(ee_pos - goal_ee_pos)
        
        # 可视化这一步
        start_ee, _ = self._get_current_ee_pose(current_config)
        end_ee, _ = self._get_current_ee_pose(q_new.tolist())
        
        debug_id = p.addUserDebugLine(
            start_ee, end_ee, [1, 0, 1], 2, 0  # 紫色线表示势场路径
        )
        
        self.debug_lines.append(debug_id)
        
        return q_new.tolist(), cost
    
    def clear_visualization(self) -> None:
        """清除路径可视化"""
        for debug_id in self.debug_lines:
            p.removeUserDebugItem(debug_id)
        self.debug_lines = []
    
    def set_reference_path(self, reference_path: List[List[float]]) -> None:
        """设置参考路径（用于RRT*和势场法结合）
        
        Args:
            reference_path: RRT*生成的参考路径
        """
        self.reference_path = reference_path
        print(f"设置参考路径，包含 {len(reference_path)} 个点")
    
    