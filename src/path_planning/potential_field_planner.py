# import numpy as np
# import pybullet as p
# import random
# import time
# from scipy.spatial import KDTree
# from typing import List, Tuple, Dict, Optional, Any, Callable

# class PotentialFieldPlanner:
#     """基于势场（Potential Field）的机械臂路径规划示例。

#     在关节空间中迭代，但对目标的吸引力使用 3D（XYZ），
#     对障碍物的排斥力仅考虑 XY 平面的距离，视障碍物为在 Z 方向无限延伸的圆柱。

#     Args:
#         robot_id: PyBullet 中的机器人 ID
#         joint_indices: 需要控制的关节索引列表
#         lower_limits: 各个关节的最小角度
#         upper_limits: 各个关节的最大角度
#         ee_link_index: 末端执行器 link 的索引
#         obstacle_tracker: 用于获取障碍物信息的实例
#         max_iterations: 迭代次数上限
#         step_size: 每次梯度下降时更新关节角的步长
#         d0: 排斥势作用的有效距离（排斥生效范围）
#         K_att: 吸引势增益系数
#         K_rep: 排斥势增益系数
#         goal_threshold: 判断到达目标的距离阈值（以末端执行器的笛卡尔距离衡量）
#         collision_check_step: 用于插值碰撞检测的步长
#     """
#     def __init__(
#         self,
#         robot_id: int,
#         joint_indices: List[int],
#         lower_limits: List[float],
#         upper_limits: List[float],
#         ee_link_index: int,
#         obstacle_tracker: Any,
#         max_iterations: int = 300,
#         step_size: float = 0.01,
#         d0: float = 0.2,
#         K_att: float = 1.0,
#         K_rep: float = 1.0,
#         goal_threshold: float = 0.05,
#         collision_check_step: float = 0.05
#     ):
#         self.robot_id = robot_id
#         self.joint_indices = joint_indices
#         self.lower_limits = lower_limits
#         self.upper_limits = upper_limits
#         self.ee_link_index = ee_link_index
#         self.obstacle_tracker = obstacle_tracker
        
#         self.max_iterations = max_iterations
#         self.step_size = step_size
#         self.d0 = d0
#         self.K_att = K_att
#         self.K_rep = K_rep
#         self.goal_threshold = goal_threshold
#         self.collision_check_step = collision_check_step
        
#         self.dimension = len(joint_indices)
    
#     def _get_current_ee_pose(self, joint_positions: List[float]) -> Tuple[np.ndarray, np.ndarray]:
#         """获取给定关节角时末端执行器的位姿（位置、四元数）。"""
#         # 记录当前的关节角状态
#         current_states = [p.getJointState(self.robot_id, i)[0] for i in self.joint_indices]
        
#         # 将机器人关节重置到指定状态
#         for i, idx in enumerate(self.joint_indices):
#             p.resetJointState(self.robot_id, idx, joint_positions[i])
            
#         # 获取末端位姿
#         ee_state = p.getLinkState(self.robot_id, self.ee_link_index)
#         ee_pos = np.array(ee_state[0])
#         ee_orn = np.array(ee_state[1])
        
#         # 恢复原始状态
#         for i, idx in enumerate(self.joint_indices):
#             p.resetJointState(self.robot_id, idx, current_states[i])
            
#         return ee_pos, ee_orn

#     def _is_ee_height_valid(self, joint_pos: List[float]) -> bool:
#         """判断末端执行器是否高于底座/桌面（简单判定逻辑）。"""
#         ee_pos, _ = self._get_current_ee_pose(joint_pos)
#         base_pos = p.getBasePositionAndOrientation(self.robot_id)[0]
        
#         # 假设 base_pos[2] 即桌面的高度，这里留一点冗余
#         table_height = base_pos[2]
#         min_height = table_height - 0.01
#         return ee_pos[2] > min_height

#     def _is_state_in_collision(self, joint_pos: List[float]) -> bool:
#         """检测给定关节角是否与场景中的障碍物发生碰撞（使用简单点球近似）。"""
#         # 暂存当前关节状态
#         current_states = [p.getJointState(self.robot_id, i)[0] for i in self.joint_indices]
        
#         # 重置到要检测的关节状态
#         for i, idx in enumerate(self.joint_indices):
#             p.resetJointState(self.robot_id, idx, joint_pos[i])
        
#         # 获取障碍信息
#         obstacle_states = self.obstacle_tracker.get_all_obstacle_states()
#         if obstacle_states is None or len(obstacle_states) == 0:
#             # 恢复状态
#             for i, idx in enumerate(self.joint_indices):
#                 p.resetJointState(self.robot_id, idx, current_states[i])
#             return False
        
#         # 取需要检测碰撞的部分 link
#         links_to_check = self.joint_indices + [self.ee_link_index]
        
#         collision = False
#         for link_idx in links_to_check:
#             link_state = p.getLinkState(self.robot_id, link_idx)
#             link_pos = np.array(link_state[0])
            
#             for obstacle in obstacle_states:
#                 if obstacle is None:
#                     continue
#                 obstacle_pos = obstacle['position']
#                 obstacle_radius = obstacle['radius']
                
#                 dist = np.linalg.norm(link_pos - obstacle_pos)
#                 # 简单点模型 + 一点安全裕度
#                 if dist < obstacle_radius + 0.05:
#                     collision = True
#                     break
            
#             if collision:
#                 break
        
#         # 恢复原始状态
#         for i, idx in enumerate(self.joint_indices):
#             p.resetJointState(self.robot_id, idx, current_states[i])
#         return collision
    
#     def _collision_free_interpolation(self, start_joints: List[float], end_joints: List[float]) -> bool:
#         """
#         用于检查从 start_joints 到 end_joints 线性插值的若干步内是否碰撞。
#         """
#         dist = np.linalg.norm(np.array(end_joints) - np.array(start_joints))
#         n_steps = max(2, int(dist / self.collision_check_step))
        
#         for i in range(n_steps + 1):
#             t = i / n_steps
#             interp = [s + t*(e - s) for s,e in zip(start_joints, end_joints)]
#             if (not self._is_ee_height_valid(interp)) or self._is_state_in_collision(interp):
#                 return False
#         return True
    
#     def _compute_ee_jacobian(self, joint_positions: List[float]) -> np.ndarray:
#         """
#         调用 PyBullet 的 calculateJacobian 来获取末端执行器位置的雅可比。
        
#         Returns:
#             一个 shape=(3, self.dimension) 的位置雅可比矩阵 J_pos，
#             其中 J_pos * dq = d(pos)，只考虑位置部分（忽略姿态）。
#         """
#         # 暂存状态
#         current_states = [p.getJointState(self.robot_id, i)[0] for i in self.joint_indices]
#         for i, idx in enumerate(self.joint_indices):
#             p.resetJointState(self.robot_id, idx, joint_positions[i])
        
#         zero_vec = [0.0]*self.dimension
#         jac_t, jac_r = p.calculateJacobian(
#             bodyUniqueId=self.robot_id,
#             linkIndex=self.ee_link_index,
#             localPosition=[0,0,0],
#             objPositions=joint_positions,
#             objVelocities=zero_vec,
#             objAccelerations=zero_vec
#         )
        
#         # 恢复状态
#         for i, idx in enumerate(self.joint_indices):
#             p.resetJointState(self.robot_id, idx, current_states[i])
        
#         # jac_t 为 3 x num_joints 的位置雅可比
#         J_pos = np.array(jac_t)
#         return J_pos
    
#     def _compute_potential_gradient(self, current_joints: List[float], goal_joints: List[float]) -> np.ndarray:
#         """
#         仅对障碍物计算 XY 平面上的排斥力；对目标吸引力仍计算 3D。
#         返回对关节空间的梯度 (numpy.ndarray)，供后续做关节更新。
#         """
#         # ---- 1) 获取末端执行器 3D 位置 ----
#         cur_ee_pos, _ = self._get_current_ee_pose(current_joints)   # (x, y, z)
#         goal_ee_pos, _ = self._get_current_ee_pose(goal_joints)     # (x_g, y_g, z_g)
        
#         # ---- 2) 计算 3D 吸引力 f_att ----
#         diff_to_goal = goal_ee_pos - cur_ee_pos  # shape=(3,)
#         f_att = self.K_att * diff_to_goal        # 目标吸引力，三维

#         # ---- 3) 计算 XY 平面的障碍物排斥力 f_rep_total ----
#         f_rep_total = np.zeros(3)  # 初始化为 (0, 0, 0)
        
#         obstacle_states = self.obstacle_tracker.get_all_obstacle_states()
#         if obstacle_states is not None:
#             for obs in obstacle_states:
#                 obs_pos = np.array(obs['position'])   # (x_obs, y_obs, z_obs)
#                 obs_radius = obs['radius']
                
#                 # 只考虑 x, y 计算距离：dist_xy
#                 dx = cur_ee_pos[0] - obs_pos[0]
#                 dy = cur_ee_pos[1] - obs_pos[1]
#                 dist_xy = np.sqrt(dx*dx + dy*dy)
                
#                 # 生效距离： d0 + obs_radius
#                 if dist_xy < (self.d0 + obs_radius):
#                     d_effective = dist_xy - obs_radius
#                     if d_effective < self.d0 and d_effective > 1e-6:
#                         # 避免除0
#                         rep_scale = self.K_rep * (1.0/d_effective - 1.0/self.d0) * (1.0/(d_effective**2))
                        
#                         # 只在 XY 平面上产生排斥（z 分量=0）
#                         direction_xy = np.array([dx, dy, 0.0]) / dist_xy
#                         f_rep = rep_scale * direction_xy
#                         # 累加到总排斥力
#                         f_rep_total += f_rep
        
#         # ---- 4) 合并总力 ----
#         f_total = f_att + f_rep_total  # (fx, fy, fz)

#         # ---- 5) 将末端执行器空间的力映射到关节空间: grad_q = J^T * f_total ----
#         J_pos = self._compute_ee_jacobian(current_joints)  # 3 x DOF
#         grad_q = J_pos.T.dot(f_total)  # (DOF,)
        
#         return grad_q
    
#     def _clip_joints_to_limits(self, joint_positions: np.ndarray) -> np.ndarray:
#         """把关节角裁剪到合法范围内。"""
#         clipped = []
#         for val, low, high in zip(joint_positions, self.lower_limits, self.upper_limits):
#             clipped.append(np.clip(val, low, high))
#         return np.array(clipped, dtype=float)
    
#     def plan(self, start_config: List[float], goal_config: List[float]) -> Tuple[List[List[float]], float]:
#         """
#         使用势场法从 start_config 迭代到 goal_config（更准确地说：逼近目标末端位置）。

#         Returns:
#             path: 关节空间路径（每次迭代的记录）
#             path_cost: 最终末端与目标末端位置的欧式距离（笛卡尔空间）
#         """
#         print("=== 开始基于势场的规划（仅对障碍物计算XY斥力）===")

#         # 检查起始和目标是否有效
#         if not self._is_ee_height_valid(start_config):
#             print("[警告] 起始配置末端执行器低于桌面！可能不可行。")
#         if not self._is_ee_height_valid(goal_config):
#             print("[警告] 目标配置末端执行器低于桌面！可能不可行。")

#         # 碰撞检测
#         if self._is_state_in_collision(start_config):
#             print("[错误] 起始配置碰撞！规划失败。")
#             return [], float('inf')
#         if self._is_state_in_collision(goal_config):
#             print("[错误] 目标配置碰撞！规划失败。")
#             return [], float('inf')
        
#         current_config = np.array(start_config, dtype=float)
#         path = [start_config]  # 记录迭代轨迹
        
#         for iteration in range(self.max_iterations):
#             # 计算梯度
#             grad = self._compute_potential_gradient(current_config.tolist(), goal_config)
            
#             # 进行关节更新: q_new = q_old - alpha * grad
#             # 由于上面 f_total 写法是"力"方向，为了做势能下降，这里用减号:
#             new_config = current_config - self.step_size * grad
            
#             # 裁剪到关节上下限
#             new_config = self._clip_joints_to_limits(new_config)
            
#             # 判断插值过程是否碰撞（或末端高度无效）
#             if not self._collision_free_interpolation(current_config.tolist(), new_config.tolist()):
#                 # 如果更新方向碰到了障碍物或低于桌面，简单地缩小步长再尝试
#                 half_step_config = current_config - 0.5 * self.step_size * grad
#                 half_step_config = self._clip_joints_to_limits(half_step_config)
                
#                 if not self._collision_free_interpolation(current_config.tolist(), half_step_config.tolist()):
#                     # 彻底动不了，说明陷入局部极小值或不可行
#                     print(f"[信息] 第 {iteration} 次迭代：被困在局部极小值附近，无法前进。")
#                     break
#                 else:
#                     current_config = half_step_config
#             else:
#                 current_config = new_config
            
#             path.append(current_config.tolist())
            
#             # 判断末端执行器与目标末端执行器的距离
#             cur_ee_pos, _ = self._get_current_ee_pose(current_config.tolist())
#             goal_ee_pos, _ = self._get_current_ee_pose(goal_config)
#             dist_to_goal = np.linalg.norm(cur_ee_pos - goal_ee_pos)
            
#             if iteration % 50 == 0:
#                 print(f" 迭代 {iteration}/{self.max_iterations}，当前末端 -> 目标距离：{dist_to_goal:.3f}")
            
#             if dist_to_goal < self.goal_threshold:
#                 print(f"在第 {iteration} 次迭代成功到达目标附近，末端距离：{dist_to_goal:.3f}")
#                 break
        
#         # 计算最终末端 -> 目标末端的距离（作为返回 cost）
#         final_ee_pos, _ = self._get_current_ee_pose(current_config.tolist())
#         goal_ee_pos, _ = self._get_current_ee_pose(goal_config)
#         path_cost = np.linalg.norm(final_ee_pos - goal_ee_pos)
        
#         print("=== 势场法规划结束 ===")
#         return path, path_cost
    
#     def clear_visualization(self) -> None:
#         """如果之前需要清除可视化，这里可自定义处理；目前留空。"""
#         pass
    
#     def generate_smooth_trajectory(self, path: List[List[float]], smoothing_steps: int = 10) -> List[List[float]]:
#         """
#         对离散的 path 做平滑插值。
#         """
#         if not path or len(path) < 2:
#             return path
        
#         smooth_trajectory = []
#         for i in range(len(path) - 1):
#             start = path[i]
#             end = path[i + 1]
#             for step in range(smoothing_steps + 1):
#                 t = step / smoothing_steps
#                 interpolated = [s + t*(e - s) for s,e in zip(start, end)]
#                 smooth_trajectory.append(interpolated)
        
#         return smooth_trajectory
