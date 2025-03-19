import numpy as np
import pybullet as p
from typing import List, Tuple, Optional

class TrajectoryPlanner:
    """
    轨迹规划器，用于生成机器人移动轨迹
    """
    
    @staticmethod
    def generate_joint_trajectory(start_joints: List[float], end_joints: List[float], steps: int = 100) -> List[List[float]]:
        """
        生成从起始到结束关节位置的平滑轨迹
        
        参数:
        start_joints: 起始关节位置
        end_joints: 结束关节位置
        steps: 插值步数
        
        返回:
        trajectory: 关节位置列表
        """
        trajectory = []
        for step in range(steps + 1):
            t = step / steps  # 归一化步长
            # 线性插值
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
        在笛卡尔空间中生成线性轨迹
        
        参数:
        robot_id: 机器人ID
        arm_idx: 机器人手臂关节索引列表
        ee_idx: 末端执行器索引
        start_joints: 起始关节位置
        target_pos: 目标位置
        target_orn: 目标方向
        steps: 插值步数
        
        返回:
        trajectory: 关节位置列表
        """
        # 设置起始位置
        for i, joint_idx in enumerate(arm_idx):
            p.resetJointState(robot_id, joint_idx, start_joints[i])
        
        # 获取当前末端执行器姿态
        ee_state = p.getLinkState(robot_id, ee_idx)
        start_pos = np.array(ee_state[0])
        
        # 生成线性轨迹
        trajectory = []
        
        # 初始化IK求解器
        from src.ik_solver import DifferentialIKSolver
        ik_solver = DifferentialIKSolver(robot_id, ee_idx, damping=0.05)
        
        for step in range(steps + 1):
            t = step / steps  # 归一化步长
            
            # 线性插值
            pos = start_pos + t * (target_pos - start_pos)
            
            # 解算当前笛卡尔位置的IK
            current_joints = ik_solver.solve(pos, target_orn, start_joints, max_iters=50, tolerance=0.001)
            
            # 将解决方案添加到轨迹
            trajectory.append(current_joints)
            
            # 重置到起始位置
            for i, joint_idx in enumerate(arm_idx):
                p.resetJointState(robot_id, joint_idx, start_joints[i])
        
        return trajectory
