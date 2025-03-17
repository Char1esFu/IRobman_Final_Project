import numpy as np
import pybullet as p
import time
from typing import List, Tuple, Dict, Optional, Any

class PotentialFieldPlanner:
    """
    基于势场法的局部路径规划器，用于实时避开障碍物
    
    参数:
        robot_id: PyBullet机器人ID
        joint_indices: 控制的关节索引列表
        ee_link_index: 末端执行器链接索引
        obstacle_tracker: 障碍物追踪器实例
        attractive_gain: 引力增益系数
        repulsive_gain: 斥力增益系数
        max_attractive_force: 最大引力
        influence_radius: 障碍物影响半径
        safety_distance: 与障碍物的安全距离
        max_step_size: 最大步长
        damping: 阻尼系数，用于平滑运动
    """
    def __init__(
        self,
        robot_id: int,
        joint_indices: List[int],
        ee_link_index: int,
        obstacle_tracker: Any,
        attractive_gain: float = 5.0,
        repulsive_gain: float = 20.0,
        max_attractive_force: float = 0.5,
        influence_radius: float = 0.5,
        safety_distance: float = 0.2,
        max_step_size: float = 0.05,
        damping: float = 0.01
    ):
        self.robot_id = robot_id
        self.joint_indices = joint_indices
        self.ee_link_index = ee_link_index
        self.obstacle_tracker = obstacle_tracker
        
        # 势场参数
        self.attractive_gain = attractive_gain
        self.repulsive_gain = repulsive_gain
        self.max_attractive_force = max_attractive_force
        self.influence_radius = influence_radius
        self.safety_distance = safety_distance
        self.max_step_size = max_step_size
        self.damping = damping
        
        # 可视化调试线条ID
        self.debug_lines = []
        
    def _get_current_ee_pose(self, joint_positions: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取给定关节位置下的末端执行器位姿
        
        参数:
            joint_positions: 关节位置列表
            
        返回:
            末端执行器位置和方向
        """
        # 保存当前关节状态
        current_states = []
        for i, joint_idx in enumerate(self.joint_indices):
            current_states.append(p.getJointState(self.robot_id, joint_idx)[0])
        
        # 临时设置关节位置
        for i, joint_idx in enumerate(self.joint_indices):
            p.resetJointState(self.robot_id, joint_idx, joint_positions[i])
        
        # 获取末端执行器位姿
        ee_state = p.getLinkState(self.robot_id, self.ee_link_index)
        ee_position = np.array(ee_state[0])
        ee_orientation = np.array(ee_state[1])
        
        # 恢复原始关节状态
        for i, joint_idx in enumerate(self.joint_indices):
            p.resetJointState(self.robot_id, joint_idx, current_states[i])
        
        return ee_position, ee_orientation
    
    def _calculate_attractive_force(self, current_pos: np.ndarray, goal_pos: np.ndarray) -> np.ndarray:
        """
        计算引力（吸引力）
        
        参数:
            current_pos: 当前位置
            goal_pos: 目标位置
            
        返回:
            引力向量
        """
        # 计算方向向量
        direction = goal_pos - current_pos
        distance = np.linalg.norm(direction)
        
        # 如果距离为零，返回零向量
        if distance < 1e-6:
            return np.zeros(3)
        
        # 归一化方向向量
        direction = direction / distance
        
        # 计算引力大小（线性引力场）
        magnitude = min(self.attractive_gain * distance, self.max_attractive_force)
        
        # 返回引力向量
        return magnitude * direction
    
    def _calculate_repulsive_force(self, current_pos: np.ndarray, obstacle_positions: List[np.ndarray], 
                                  obstacle_radii: List[float]) -> np.ndarray:
        """
        计算斥力（排斥力）
        
        参数:
            current_pos: 当前位置
            obstacle_positions: 障碍物位置列表
            obstacle_radii: 障碍物半径列表
            
        返回:
            斥力向量
        """
        total_repulsive_force = np.zeros(3)
        
        for obstacle_pos, obstacle_radius in zip(obstacle_positions, obstacle_radii):
            # 计算到障碍物的向量
            to_obstacle = obstacle_pos - current_pos
            distance = np.linalg.norm(to_obstacle)
            
            # 考虑障碍物半径
            distance_to_surface = max(0.001, distance - obstacle_radius)
            
            # 如果在影响半径内，计算斥力
            if distance_to_surface < self.influence_radius:
                # 归一化方向向量（远离障碍物）
                if distance < 1e-6:
                    # 如果距离太小，选择一个随机方向
                    direction = np.random.randn(3)
                    direction = direction / np.linalg.norm(direction)
                else:
                    direction = -to_obstacle / distance
                
                # 计算斥力大小（反比于距离的平方）
                if distance_to_surface <= self.safety_distance:
                    # 如果太接近障碍物，使用更大的斥力
                    magnitude = self.repulsive_gain * (1.0 / self.safety_distance - 1.0 / self.influence_radius) * (1.0 / distance_to_surface) ** 2
                else:
                    magnitude = self.repulsive_gain * (1.0 / distance_to_surface - 1.0 / self.influence_radius) * (1.0 / distance_to_surface) ** 2
                
                # 累加斥力
                total_repulsive_force += magnitude * direction
        
        return total_repulsive_force
    
    def _calculate_jacobian(self, joint_positions: List[float]) -> np.ndarray:
        """
        计算雅可比矩阵
        
        参数:
            joint_positions: 关节位置列表
            
        返回:
            雅可比矩阵 (3 x n)，其中n是关节数量
        """
        # 保存当前关节状态
        current_states = []
        for i, joint_idx in enumerate(self.joint_indices):
            current_states.append(p.getJointState(self.robot_id, joint_idx)[0])
        
        # 临时设置关节位置
        for i, joint_idx in enumerate(self.joint_indices):
            p.resetJointState(self.robot_id, joint_idx, joint_positions[i])
        
        # 计算雅可比矩阵
        linear_jacobian = []
        for i, joint_idx in enumerate(self.joint_indices):
            # 计算关节轴方向和位置
            joint_info = p.getJointInfo(self.robot_id, joint_idx)
            joint_pos = np.array(p.getLinkState(self.robot_id, joint_info[0])[0])
            
            # 获取关节轴方向（在世界坐标系中）
            if joint_info[2] == p.JOINT_REVOLUTE:
                # 旋转关节
                joint_axis = np.array(joint_info[13])
                # 获取末端执行器位置
                ee_pos = np.array(p.getLinkState(self.robot_id, self.ee_link_index)[0])
                # 计算叉积
                linear_jac_col = np.cross(joint_axis, ee_pos - joint_pos)
                linear_jacobian.append(linear_jac_col)
            elif joint_info[2] == p.JOINT_PRISMATIC:
                # 移动关节
                joint_axis = np.array(joint_info[13])
                linear_jacobian.append(joint_axis)
            else:
                # 其他类型的关节
                linear_jacobian.append(np.zeros(3))
        
        # 恢复原始关节状态
        for i, joint_idx in enumerate(self.joint_indices):
            p.resetJointState(self.robot_id, joint_idx, current_states[i])
        
        return np.array(linear_jacobian).T  # 转置为 3 x n
    
    def _calculate_joint_velocities(self, joint_positions: List[float], force: np.ndarray) -> np.ndarray:
        """
        使用雅可比矩阵计算关节速度
        
        参数:
            joint_positions: 当前关节位置
            force: 笛卡尔空间中的力向量
            
        返回:
            关节速度数组
        """
        # 计算雅可比矩阵
        jacobian = self._calculate_jacobian(joint_positions)
        
        # 使用伪逆计算关节速度
        # J+ = J^T * (J * J^T)^-1
        j_transpose = jacobian.T
        
        # 添加阻尼以处理奇异点
        lambda_squared = self.damping ** 2
        identity = np.eye(jacobian.shape[0])
        
        # 阻尼最小二乘法
        j_pseudo_inv = j_transpose @ np.linalg.inv(jacobian @ j_transpose + lambda_squared * identity)
        
        # 计算关节速度
        joint_velocities = j_pseudo_inv @ force
        
        return joint_velocities
    
    def _limit_step_size(self, joint_velocities: np.ndarray) -> np.ndarray:
        """
        限制关节速度步长
        
        参数:
            joint_velocities: 关节速度数组
            
        返回:
            限制后的关节速度数组
        """
        # 计算速度范数
        velocity_norm = np.linalg.norm(joint_velocities)
        
        # 如果速度超过最大步长，进行缩放
        if velocity_norm > self.max_step_size and velocity_norm > 1e-6:
            joint_velocities = joint_velocities * (self.max_step_size / velocity_norm)
        
        return joint_velocities
    
    def _visualize_forces(self, position: np.ndarray, attractive_force: np.ndarray, 
                         repulsive_force: np.ndarray, total_force: np.ndarray) -> None:
        """
        可视化力向量
        
        参数:
            position: 当前位置
            attractive_force: 引力向量
            repulsive_force: 斥力向量
            total_force: 合力向量
        """
        # 清除之前的可视化线条
        for line_id in self.debug_lines:
            p.removeUserDebugItem(line_id)
        self.debug_lines = []
        
        # 可视化引力（绿色）
        scale = 0.2  # 缩放因子，使向量在可视化中更明显
        if np.linalg.norm(attractive_force) > 1e-6:
            line_id = p.addUserDebugLine(
                position,
                position + scale * attractive_force,
                [0, 1, 0],  # 绿色
                2,
                0
            )
            self.debug_lines.append(line_id)
        
        # 可视化斥力（红色）
        if np.linalg.norm(repulsive_force) > 1e-6:
            line_id = p.addUserDebugLine(
                position,
                position + scale * repulsive_force,
                [1, 0, 0],  # 红色
                2,
                0
            )
            self.debug_lines.append(line_id)
        
        # 可视化合力（蓝色）
        if np.linalg.norm(total_force) > 1e-6:
            line_id = p.addUserDebugLine(
                position,
                position + scale * total_force,
                [0, 0, 1],  # 蓝色
                3,
                0
            )
            self.debug_lines.append(line_id)
    
    def plan_next_step(self, current_joints: List[float], goal_pos: np.ndarray, 
                      visualize: bool = True) -> List[float]:
        """
        规划下一步的关节位置
        
        参数:
            current_joints: 当前关节位置
            goal_pos: 目标位置
            visualize: 是否可视化力向量
            
        返回:
            下一步的关节位置
        """
        # 获取当前末端执行器位置
        current_pos, _ = self._get_current_ee_pose(current_joints)
        
        # 获取障碍物位置和半径
        obstacle_positions = []
        obstacle_radii = []
        
        # 从障碍物追踪器获取障碍物信息
        obstacle_states = self.obstacle_tracker.get_all_obstacle_states()
        for i, state in enumerate(obstacle_states):
            if state is not None:
                # 正确访问字典中的position字段
                position = state['position']  # 获取位置数组
                radius = state['radius']  # 获取半径
                obstacle_positions.append(position)
                obstacle_radii.append(radius)
        
        # 计算引力
        attractive_force = self._calculate_attractive_force(current_pos, goal_pos)
        
        # 计算斥力
        repulsive_force = self._calculate_repulsive_force(current_pos, obstacle_positions, obstacle_radii)
        
        # 合并力
        total_force = attractive_force + repulsive_force
        
        # 可视化力向量
        if visualize:
            self._visualize_forces(current_pos, attractive_force, repulsive_force, total_force)
        
        # 计算关节速度
        joint_velocities = self._calculate_joint_velocities(current_joints, total_force)
        
        # 限制步长
        joint_velocities = self._limit_step_size(joint_velocities)
        
        # 计算下一步关节位置
        next_joints = np.array(current_joints) + joint_velocities
        
        return next_joints.tolist()
    
    def refine_trajectory(self, trajectory: List[List[float]], goal_pos: np.ndarray, 
                         max_iterations: int = 100, tolerance: float = 0.01) -> List[List[float]]:
        """
        使用势场法优化轨迹，避开障碍物
        
        参数:
            trajectory: 原始轨迹（关节空间）
            goal_pos: 笛卡尔空间中的目标位置
            max_iterations: 最大迭代次数
            tolerance: 收敛容差
            
        返回:
            优化后的轨迹
        """
        if not trajectory:
            return []
        
        refined_trajectory = [trajectory[0]]  # 从起始点开始
        
        # 逐步优化轨迹中的每个点
        for i in range(1, len(trajectory)):
            current_point = refined_trajectory[-1]
            target_point = trajectory[i]
            
            # 获取当前点和目标点的末端执行器位置
            current_ee_pos, _ = self._get_current_ee_pose(current_point)
            target_ee_pos, _ = self._get_current_ee_pose(target_point)
            
            # 使用势场法从当前点移动到目标点
            intermediate_point = current_point
            for _ in range(max_iterations):
                # 计算下一步
                next_point = self.plan_next_step(intermediate_point, target_ee_pos, visualize=False)
                
                # 获取新位置
                next_ee_pos, _ = self._get_current_ee_pose(next_point)
                
                # 检查是否足够接近目标
                if np.linalg.norm(next_ee_pos - target_ee_pos) < tolerance:
                    intermediate_point = next_point
                    break
                
                intermediate_point = next_point
            
            # 添加优化后的点到轨迹中
            refined_trajectory.append(intermediate_point)
        
        return refined_trajectory
    
    def execute_trajectory_with_avoidance(self, sim, trajectory: List[List[float]], 
                                         goal_pos: np.ndarray, dt: float = 1/240.0,
                                         visualize_obstacles: bool = False) -> None:
        """
        执行轨迹，同时实时避开障碍物
        
        参数:
            sim: 仿真环境
            trajectory: 关节空间轨迹
            goal_pos: 笛卡尔空间中的目标位置
            dt: 时间步长
            visualize_obstacles: 是否可视化障碍物边界框
        """
        if not trajectory:
            print("轨迹为空，无法执行")
            return
        
        print("开始执行轨迹，带有实时避障...")
        
        # 从轨迹起点开始
        current_joints = trajectory[0]
        
        # 设置机器人到起始位置
        for i, joint_idx in enumerate(self.joint_indices):
            p.resetJointState(self.robot_id, joint_idx, current_joints[i])
        
        # 用于存储边界框可视化ID
        bounding_box_ids = []
        
        # 设置最大执行时间（秒）和开始时间
        max_execution_time = 600.0  # 最大执行600秒
        start_time = time.time()
        
        # 设置到达目标的阈值
        goal_threshold = 0.05  # 5厘米
        
        # 设置最大迭代次数，防止无限循环
        max_iterations = 2000
        iteration = 0
        
        # 获取初始末端执行器位置
        current_ee_pos, _ = self._get_current_ee_pose(current_joints)
        initial_distance = np.linalg.norm(current_ee_pos - goal_pos)
        closest_distance = initial_distance
        
        # 轨迹索引
        trajectory_index = 1
        
        # 设置轨迹点切换距离阈值
        waypoint_threshold = 0.1  # 10厘米
        
        # 设置障碍物避让模式的标志
        avoiding_obstacle = False
        
        # 主循环：直到到达目标位置或超时
        while True:
            # 检查是否超时
            if time.time() - start_time > max_execution_time:
                print(f"执行超时（{max_execution_time}秒），终止执行")
                break
                
            # 检查是否达到最大迭代次数
            if iteration >= max_iterations:
                print(f"达到最大迭代次数（{max_iterations}），终止执行")
                break
                
            # 更新障碍物跟踪
            rgb_static, depth_static, seg_static = sim.get_static_renders()
            detections = self.obstacle_tracker.detect_obstacles(rgb_static, depth_static, seg_static)
            tracked_positions = self.obstacle_tracker.update(detections)
            
            # 可视化跟踪的障碍物（如果需要）
            if visualize_obstacles:
                # 清除之前的边界框
                for debug_line in bounding_box_ids:
                    p.removeUserDebugItem(debug_line)
                # 绘制新的边界框
                bounding_box_ids = self.obstacle_tracker.visualize_tracking_3d(tracked_positions)
            
            # 获取当前末端执行器位置
            current_ee_pos, _ = self._get_current_ee_pose(current_joints)
            
            # 计算到目标的距离
            distance_to_goal = np.linalg.norm(current_ee_pos - goal_pos)
            
            # 更新最近距离
            if distance_to_goal < closest_distance:
                closest_distance = distance_to_goal
            
            # 检查是否到达目标
            if distance_to_goal < goal_threshold:
                print(f"到达目标位置！距离: {distance_to_goal:.4f}m")
                break
            
            # 获取轨迹中的下一个目标点
            if trajectory_index < len(trajectory):
                target_joints = trajectory[trajectory_index]
                target_ee_pos, _ = self._get_current_ee_pose(target_joints)
                
                # 计算到下一个轨迹点的距离
                distance_to_waypoint = np.linalg.norm(current_ee_pos - target_ee_pos)
                
                # 检查是否有障碍物
                obstacle_states = self.obstacle_tracker.get_all_obstacle_states()
                obstacles_detected = False
                
                for state in obstacle_states:
                    if state is not None:
                        position = state['position']
                        radius = state['radius']
                        
                        # 计算障碍物到当前位置和目标位置之间线段的最短距离
                        obstacle_distance = self._point_to_line_segment_distance(
                            position, current_ee_pos, target_ee_pos)
                        
                        # 如果障碍物太近，进入避障模式
                        if obstacle_distance < (radius + self.safety_distance):
                            obstacles_detected = True
                            avoiding_obstacle = True
                            break
                
                # 如果没有障碍物或者已经足够接近下一个轨迹点，前进到下一个点
                if (not obstacles_detected or distance_to_waypoint < waypoint_threshold) and avoiding_obstacle == False:
                    trajectory_index += 1
                    print(f"前进到轨迹点 {trajectory_index}/{len(trajectory)}")
            else:
                # 如果已经用完轨迹点，直接使用最终目标位置
                target_ee_pos = goal_pos
            
            # 使用势场法计算避障路径
            next_joints = self.plan_next_step(current_joints, target_ee_pos, visualize=True)
            
            # 移动机器人
            sim.robot.position_control(next_joints)
            
            # 更新仿真
            for _ in range(1):
                sim.step()
                time.sleep(dt)
            
            # 更新当前关节位置
            current_joints = sim.robot.get_joint_positions()
            
            # 每100次迭代打印一次进度
            if iteration % 100 == 0:
                print(f"迭代 {iteration}, 距离目标: {distance_to_goal:.4f}m, 最近距离: {closest_distance:.4f}m")
                
            iteration += 1
        
        # 清除所有边界框
        if visualize_obstacles and bounding_box_ids:
            for debug_line in bounding_box_ids:
                p.removeUserDebugItem(debug_line)
        
        # 清除力向量可视化
        for line_id in self.debug_lines:
            p.removeUserDebugItem(line_id)
        self.debug_lines = []
        
        print("轨迹执行完成")
    
    def _point_to_line_segment_distance(self, point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray) -> float:
        """
        计算点到线段的最短距离
        
        参数:
            point: 点坐标
            line_start: 线段起点
            line_end: 线段终点
            
        返回:
            点到线段的最短距离
        """
        # 线段向量
        line_vec = line_end - line_start
        line_length = np.linalg.norm(line_vec)
        
        # 如果线段长度为零，直接返回点到起点的距离
        if line_length < 1e-6:
            return np.linalg.norm(point - line_start)
        
        # 归一化线段向量
        line_vec_normalized = line_vec / line_length
        
        # 计算点到线段起点的向量
        point_vec = point - line_start
        
        # 计算点在线段上的投影长度
        projection_length = np.dot(point_vec, line_vec_normalized)
        
        # 如果投影在线段外部，返回点到最近端点的距离
        if projection_length < 0:
            return np.linalg.norm(point - line_start)
        elif projection_length > line_length:
            return np.linalg.norm(point - line_end)
        
        # 计算投影点
        projection_point = line_start + projection_length * line_vec_normalized
        
        # 返回点到投影点的距离
        return np.linalg.norm(point - projection_point) 