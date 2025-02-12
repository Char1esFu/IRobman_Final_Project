import pybullet as p
import pybullet_data
import numpy as np
from math import pi
import time

class DifferentialIKSolver:
    def __init__(self, robot_id, ee_link_index, damping=0.001, use_shadow=True,
                 urdf_path="franka_panda/panda.urdf", base_position=[0, 0, 0], use_fixed_base=True):
        """
        如果 use_shadow 为 True，则在一个独立的 DIRECT 客户端中加载一个影子机器人进行计算，
        这样就不会对主仿真造成任何干扰。
        """
        self.damping = damping
        self.ee_link_index = ee_link_index
        self.use_shadow = use_shadow
        if self.use_shadow:
            # 创建独立的物理客户端（DIRECT 模式，不显示界面）
            self.cid = p.connect(p.DIRECT)
            p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.cid)
            # 加载影子机器人
            self.robot_id = p.loadURDF(urdf_path, base_position, useFixedBase=use_fixed_base,
                                       physicsClientId=self.cid)
        else:
            # 使用传入的 robot_id（主仿真中的机器人）
            self.robot_id = robot_id
            self.cid = 0  # 默认客户端 id
        
        # 获取受控关节索引（这里只选 REVOLUTE 类型的关节，可根据需要扩展）
        self.joint_indices = []
        num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.cid)
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, i, physicsClientId=self.cid)
            if joint_info[2] == p.JOINT_REVOLUTE:
                self.joint_indices.append(i)
        self.num_joints = len(self.joint_indices)
        print(f"\nRobot Configuration (shadow: {self.use_shadow}):")
        print(f"Number of controlled joints: {self.num_joints}")
        print(f"Joint indices: {self.joint_indices}")

    def get_current_ee_pose(self):
        ee_state = p.getLinkState(self.robot_id, self.ee_link_index, physicsClientId=self.cid)
        return np.array(ee_state[0]), np.array(ee_state[1])

    def get_jacobian(self, joint_positions):
        delta = 1e-3  # 数值微分步长
        jac = np.zeros((6, self.num_joints))  # 6 x n 雅可比矩阵
        
        # 保存原始状态
        original_pos = joint_positions.copy()
        current_pos, current_orn = self.get_current_ee_pose()
        
        for i in range(self.num_joints):
            joint_positions = original_pos.copy()
            joint_positions[i] += delta
            
            # 在影子机器人中设置新的关节状态
            for idx, pos in zip(self.joint_indices, joint_positions):
                p.resetJointState(self.robot_id, idx, pos, physicsClientId=self.cid)
            
            # 计算新的末端位姿
            new_pos, new_orn = self.get_current_ee_pose()
            
            # 位置雅可比
            jac[:3, i] = (new_pos - current_pos) / delta
            
            # 姿态雅可比（通过 quaternion 差分估计角速度）
            orn_diff = p.getDifferenceQuaternion(current_orn.tolist(), new_orn.tolist())
            jac[3:, i] = np.array(orn_diff[:3]) / delta
        
        # 恢复原始状态
        for idx, pos in zip(self.joint_indices, original_pos):
            p.resetJointState(self.robot_id, idx, pos, physicsClientId=self.cid)
            
        return jac

    def solve(self, target_pos, target_orn, current_joint_positions, max_iters=50, tolerance=1e-2):
        """
        求解 IK。当前计算完全在影子机器人上进行，不会影响主仿真中的机器人状态。
        """
        current_joints = np.array(current_joint_positions)
        
        for iter in range(max_iters):
            current_pos, current_orn = self.get_current_ee_pose()
            pos_error = target_pos - current_pos
            pos_error_norm = np.linalg.norm(pos_error)
            
            orn_error = np.array(p.getDifferenceQuaternion(current_orn.tolist(), target_orn)[:3])
            orn_error_norm = np.linalg.norm(orn_error)
            
            # 合并位置与姿态误差
            error = np.concatenate([pos_error, orn_error])
            print(f"Iteration {iter}, Pos Error: {pos_error_norm:.6f}, Ori Error: {orn_error_norm:.6f}")
            
            if pos_error_norm < tolerance and orn_error_norm < tolerance:
                print("IK solved successfully!")
                break
            
            J = self.get_jacobian(current_joints)
            
            # 阻尼最小二乘法求解 delta_q
            delta_q = np.linalg.solve(
                J.T @ J + self.damping * np.eye(self.num_joints),
                J.T @ error
            )
            
            # 更新关节角度
            current_joints += delta_q
            
            # 在影子机器人中更新状态
            for i, joint_idx in enumerate(self.joint_indices):
                p.resetJointState(self.robot_id, joint_idx, current_joints[i], physicsClientId=self.cid)
                
        return current_joints.tolist()
    
    def disconnect(self):
        if self.use_shadow:
            p.disconnect(self.cid)

# 示例测试函数（不影响主仿真）
def test_ik_solver_shadow():
    # 这里我们启动主仿真（GUI 模式），但 IK 求解将在影子客户端中计算
    main_cid = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    # 注意：主仿真中的机器人状态不会受到影子客户端中 IK 计算的影响
    main_robot_id = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)
    
    # 设定末端 link index（这里仍用 7 作为例子，具体请根据实际情况设定）
    ee_link_index = 7
    
    # 获取主仿真中机械臂的初始关节角（这里只取前 7 个 REVOLUTE 关节）
    initial_joints = [0, -pi/4, 0, -pi/2, 0, pi/3, 0]
    for i, joint_idx in enumerate(range(7)):  # 假设前 7 个关节为机械臂
        p.resetJointState(main_robot_id, joint_idx, initial_joints[i])
    
    # 定义目标末端位姿
    target_pos = np.array([0.5, 0.2, 0.7])
    target_orn = p.getQuaternionFromEuler([0, pi/2, 0])
    
    print("\nStarting IK solution in shadow simulation...")
    
    # 创建 IK 求解器实例，开启 use_shadow=True
    ik_solver = DifferentialIKSolver(
        robot_id=main_robot_id,      # 传入主仿真中的机器人 id，仅用于参数参考
        ee_link_index=ee_link_index,
        damping=0.05,
        use_shadow=True,             # 关键：在独立的影子客户端中计算
        urdf_path="franka_panda/panda.urdf",
        base_position=[0, 0, 0],
        use_fixed_base=True
    )
    
    # 计算 IK（使用初始状态作为起始状态）
    new_joints = ik_solver.solve(target_pos, target_orn, initial_joints, max_iters=50, tolerance=0.01)
    print("\nFinal joint angles (IK solution):", new_joints)
    
    # 断开影子客户端
    ik_solver.disconnect()
    
    # 在主仿真中，应用 IK 解（仅对显示和后续规划有用）
    for i, joint_idx in enumerate(range(7)):
        p.setJointMotorControl2(main_robot_id, joint_idx, p.POSITION_CONTROL, new_joints[i])
    
    # 运行一段时间查看结果
    for _ in range(1000):
        p.stepSimulation()
        time.sleep(1./240.)
    p.disconnect(main_cid)

if __name__ == "__main__":
    test_ik_solver_shadow()
