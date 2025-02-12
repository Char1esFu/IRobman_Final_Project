# ./src/ik_solver.py
import pybullet as p
import pybullet_data
import numpy as np
from math import pi
import time

class DifferentialIKSolver:
    def __init__(self, robot_id, ee_link_index, damping=0.001):
        self.robot_id = robot_id
        self.ee_link_index = ee_link_index
        self.damping = damping
        
        # 采集所有可控关节：同时包括 REVOLUTE 和 PRISMATIC 类型
        self.joint_indices = []
        for i in range(p.getNumJoints(robot_id)):
            joint_info = p.getJointInfo(robot_id, i)
            if joint_info[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                self.joint_indices.append(i)
        self.num_joints = len(self.joint_indices)
        print(f"\nRobot Configuration:")
        print(f"Number of controlled joints: {self.num_joints}")
        print(f"Joint indices: {self.joint_indices}")

    def get_current_ee_pose(self):
        ee_state = p.getLinkState(self.robot_id, self.ee_link_index)
        return np.array(ee_state[0]), np.array(ee_state[1])

    def get_jacobian(self, joint_positions):
        # 使用 pybullet.calculateJacobian 直接计算雅可比矩阵
        local_position = [0, 0, 0]  # 在 link 坐标系中的目标点，通常选用 [0,0,0]
        zero_vec = [0.0] * self.num_joints  # 对应关节速度和加速度全零
        jac_linear, jac_angular = p.calculateJacobian(
            self.robot_id,
            self.ee_link_index,
            local_position,
            joint_positions,
            zero_vec,
            zero_vec
        )
        J_linear = np.array(jac_linear)    # shape: 3 x n
        J_angular = np.array(jac_angular)  # shape: 3 x n
        return np.vstack((J_linear, J_angular))  # 6 x n

    def solve(self, target_pos, target_orn, current_joint_positions, max_iters=50, tolerance=1e-2):
        """IK 求解：利用阻尼最小二乘法"""
        current_joints = np.array(current_joint_positions)
        
        for iter in range(max_iters):
            current_pos, current_orn = self.get_current_ee_pose()
            
            pos_error = target_pos - current_pos
            pos_error_norm = np.linalg.norm(pos_error)
            
            orn_error = np.array(p.getDifferenceQuaternion(current_orn.tolist(), target_orn)[:3])
            orn_error_norm = np.linalg.norm(orn_error)
            
            # 组合位置和姿态误差
            error = np.concatenate([pos_error, orn_error])
            print(f"Iteration {iter}, Position Error: {pos_error_norm:.6f}, Orientation Error: {orn_error_norm:.6f}")
            
            if pos_error_norm < tolerance and orn_error_norm < tolerance:
                print("IK solved successfully!")
                break
            
            J = self.get_jacobian(current_joints.tolist())
            
            # 阻尼最小二乘求解关节变化量
            delta_q = np.linalg.solve(
                J.T @ J + self.damping * np.eye(self.num_joints),
                J.T @ error
            )
            
            # 更新关节角度
            current_joints += delta_q
            
            # 应用更新后的关节状态
            for i, joint_idx in enumerate(self.joint_indices):
                p.resetJointState(self.robot_id, joint_idx, current_joints[i])
                
        return current_joints.tolist()

def test_ik_solver():
    # 初始化 pybullet
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    p.loadURDF("plane.urdf")
    robot_id = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)
    
    # 修改末端 link index，包含抓取关节
    ee_link_index = 11
    
    ik_solver = DifferentialIKSolver(robot_id, ee_link_index)
    
    # 初始化关节角度：假设前 7 个为机械臂角度，其余为抓取关节（这里默认设置为 0）
    initial_joints = [0, -pi/4, 0, -pi/2, 0, pi/3, 0] + [0.04] * (ik_solver.num_joints - 7)
    for i, joint_idx in enumerate(ik_solver.joint_indices):
        p.resetJointState(robot_id, joint_idx, initial_joints[i])
    
    # 目标末端位姿
    target_pos = np.array([0.5, 0.2, 0.7])
    target_orn = p.getQuaternionFromEuler([0, pi/2, 0])
    
    print("\nStarting IK solution...")
    print(f"Target position: {target_pos}")
    print(f"Target orientation: {target_orn}")
    
    try:
        new_joints = ik_solver.solve(target_pos, target_orn, initial_joints)
        print("\nFinal joint angles:", new_joints)
        
        # 应用 IK 解
        for i, joint_idx in enumerate(ik_solver.joint_indices):
            p.setJointMotorControl2(robot_id, joint_idx, p.POSITION_CONTROL, new_joints[i])
        
        # 保持仿真运行一段时间
        for _ in range(1000):
            p.stepSimulation()
            time.sleep(1./240.)
            
    except Exception as e:
        print(f"Error during IK solution: {str(e)}")
    finally:
        p.disconnect()

if __name__ == "__main__":
    test_ik_solver()
