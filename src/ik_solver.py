import pybullet as p  # 导入 PyBullet 库
import pybullet_data  # 导入 PyBullet 自带的数据文件
import numpy as np  # 导入 NumPy，用于数值计算
from math import pi  # 从 math 库中导入 pi 常数
import time  # 导入 time 库，用于延时或计时

class DifferentialIKSolver:
    def __init__(self, robot_id, ee_link_index, damping=0.001, use_shadow_client=True):  # 定义类的初始化方法
        self.robot_id = robot_id  # 记录机器人在仿真中的 ID
        self.ee_link_index = ee_link_index  # 记录末端执行器的链接索引
        self.damping = damping  # 设置阻尼系数
        self.use_shadow_client = use_shadow_client  # 是否使用影子客户端
        
        # get robot joint index  # 获取机器人关节的索引
        self.joint_indices = []  # 存储关节索引的列表
        for i in range(p.getNumJoints(robot_id)):  # 遍历机器人所有关节
            joint_info = p.getJointInfo(robot_id, i)  # 获取关节信息
            if joint_info[2] == p.JOINT_REVOLUTE:  # 如果关节类型是旋转关节
                self.joint_indices.append(i)  # 将该关节的索引加入列表
        self.num_joints = len(self.joint_indices)  # 可控关节的数量
        
        # Create shadow client for IK calculations if needed  # 如果需要，创建影子客户端进行 IK 计算
        if self.use_shadow_client:  # 判断是否使用影子客户端
            # Create a DIRECT physics client for shadow calculations  # 创建一个 DIRECT 客户端，用于影子计算
            self.shadow_client_id = p.connect(p.DIRECT)  # 连接 PyBullet DIRECT 模式
            
            # Get the real robot's base position and orientation  # 获取真实机器人底座的位置和朝向
            robot_pos, robot_orn = p.getBasePositionAndOrientation(self.robot_id)
            
            # Load the same robot in the shadow client with the same position and orientation  # 在影子客户端加载相同的机器人，并设置相同的位姿
            p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.shadow_client_id)  # 设置 PyBullet 数据文件路径
            self.shadow_robot_id = p.loadURDF("franka_panda/panda.urdf", 
                                             basePosition=robot_pos,
                                             baseOrientation=robot_orn,
                                             useFixedBase=True, 
                                             physicsClientId=self.shadow_client_id)
            
            # Copy the current joint states from the real robot to the shadow robot  # 同步真实机器人当前关节角度到影子机器人
            for joint_idx in self.joint_indices:  # 遍历所有可控关节
                joint_state = p.getJointState(self.robot_id, joint_idx)  # 获取真实机器人关节状态
                joint_pos = joint_state[0]  # 关节位置
                joint_vel = 0.0  # 将关节速度重置为 0
                p.resetJointState(self.shadow_robot_id, joint_idx, 
                                 targetValue=joint_pos,
                                 targetVelocity=joint_vel,
                                 physicsClientId=self.shadow_client_id)
        else:  # 如果不使用影子客户端
            self.shadow_client_id = None  # 不需要影子客户端 ID
            self.shadow_robot_id = None  # 不需要影子机器人 ID
        
        print(f"\nRobot Configuration:")  # 打印机器人配置信息
        print(f"Number of controlled joints: {self.num_joints}")  # 输出控制的关节数量
        print(f"Joint indices: {self.joint_indices}")  # 输出关节索引列表
        
        if self.use_shadow_client:  # 如果使用影子客户端
            print(f"Shadow client initialized with ID: {self.shadow_client_id}")  # 打印影子客户端的 ID
            print(f"Shadow robot initialized with ID: {self.shadow_robot_id}")  # 打印影子机器人的 ID

    def get_current_ee_pose(self, use_shadow=False):  # 获取当前末端执行器的位姿
        """Get the current end effector pose"""  # 这是一段说明文档字符串
        if use_shadow and self.use_shadow_client:  # 如果指定使用影子客户端，并且影子客户端可用
            client_id = self.shadow_client_id  # 使用影子客户端的 ID
            robot_id = self.shadow_robot_id  # 使用影子机器人
        else:  # 否则
            # Use default client ID (no need to specify)  # 使用默认的客户端 ID 0
            client_id = 0  # 默认客户端 ID
            robot_id = self.robot_id  # 使用真实机器人
        
        # Switch to appropriate client  # 切换到相应的客户端（可选操作）
        p.setPhysicsEngineParameter(enableConeFriction=0, physicsClientId=client_id)  # 设置物理引擎参数
        
        ee_state = p.getLinkState(robot_id, self.ee_link_index, physicsClientId=client_id)  # 获取末端执行器的状态
        return np.array(ee_state[0]), np.array(ee_state[1])  # 返回末端执行器的位置和四元数姿态

    def get_jacobian(self, joint_positions, use_shadow=True):  # 计算雅可比矩阵
        """Calculate the Jacobian matrix"""  # 这是一段说明文档字符串
        delta = 1e-3  # numerical differentiation step  # 数值微分的步长
        jac = np.zeros((6, len(self.joint_indices)))  # 6×n jacobian  # 6×N 的雅可比矩阵，用于位置和姿态
        
        if use_shadow and self.use_shadow_client:  # 如果使用影子客户端
            client_id = self.shadow_client_id  # 影子客户端 ID
            robot_id = self.shadow_robot_id  # 影子机器人 ID
        else:  # 否则使用默认客户端
            client_id = 0  # 默认客户端 ID
            robot_id = self.robot_id  # 真实机器人 ID
        
        # Switch to appropriate client  # 切换到相应的客户端（可选操作）
        p.setPhysicsEngineParameter(enableConeFriction=0, physicsClientId=client_id)  # 设置物理引擎参数
        
        # save original position  # 保存原始关节位置
        original_pos = joint_positions.copy()
        
        # Set initial joint positions  # 设置初始关节角度
        for idx, pos in zip(self.joint_indices, original_pos):
            p.resetJointState(robot_id, idx, pos, physicsClientId=client_id)
            
        current_pos, current_orn = self.get_current_ee_pose(use_shadow=use_shadow)  # 获取当前末端执行器的位姿
        
        for i in range(len(self.joint_indices)):  # 遍历每个关节
            joint_positions = original_pos.copy()  # 复制原始关节位置
            joint_positions[i] += delta  # 仅扰动第 i 个关节
            
            # set joint state  # 设置新的关节角度
            for idx, pos in zip(self.joint_indices, joint_positions):
                p.resetJointState(robot_id, idx, pos, physicsClientId=client_id)
            
            # new position and orientation  # 获取扰动后末端执行器的新位姿
            new_pos, new_orn = self.get_current_ee_pose(use_shadow=use_shadow)
            
            # pos jacobian  # 计算位置部分的雅可比
            jac[:3, i] = (new_pos - current_pos) / delta
            
            # ori jacobian  # 计算姿态部分的雅可比
            # quaternion difference as angular velocity  # 用四元数差近似角速度
            orn_diff = p.getDifferenceQuaternion(current_orn.tolist(), new_orn.tolist(), physicsClientId=client_id)
            jac[3:, i] = np.array(orn_diff[:3]) / delta
        
        # reset joint state  # 可选：这里并没有把关节重置回去，但不影响接下来的使用
        for idx, pos in zip(self.joint_indices, original_pos):
            p.resetJointState(robot_id, idx, pos, physicsClientId=client_id)
            
        return jac  # 返回雅可比矩阵

    def solve(self, target_pos, target_orn, current_joint_positions, max_iters=50, tolerance=1e-3):  # 进行 IK 求解
        """solve IK using shadow client if available"""  # 这是一段说明文档字符串
        current_joints = np.array(current_joint_positions)  # 将当前关节角度转换为 NumPy 数组

        # 定义Franka Panda机器人的关节限位  # 七个关节对应的上下限
        joint_limits = [
            (-2.9671, 2.9671),  # Joint 1 (panda_joint1)
            (-1.8326, 1.8326),  # Joint 2 (panda_joint2)
            (-2.9671, 2.9671),  # Joint 3 (panda_joint3)
            (-3.1416, 0.0),     # Joint 4 (panda_joint4)
            (-2.9671, 2.9671),  # Joint 5 (panda_joint5)
            (-0.0873, 3.8223),  # Joint 6 (panda_joint6)
            (-2.9671, 2.9671),  # Joint 7 (panda_joint7)
        ]
        
        # If using shadow client, set initial joint positions in shadow robot  # 如果使用影子客户端，则将关节角度同步到影子机器人
        if self.use_shadow_client:
            for i, joint_idx in enumerate(self.joint_indices):
                p.resetJointState(self.shadow_robot_id, joint_idx, current_joints[i], 
                                 physicsClientId=self.shadow_client_id)
        
        for iter in range(max_iters):  # 最多迭代 max_iters 次
            # Use shadow client for calculations if available  # 如果有影子客户端，则使用它进行计算
            current_pos, current_orn = self.get_current_ee_pose(use_shadow=self.use_shadow_client)
            
            pos_error = target_pos - current_pos  # 位置误差
            pos_error_norm = np.linalg.norm(pos_error)  # 位置误差的范数
            
            client_id = self.shadow_client_id if self.use_shadow_client else 0  # Default client ID  # 选择客户端
            orn_error = np.array(p.getDifferenceQuaternion(current_orn.tolist(), target_orn, 
                                                          physicsClientId=client_id)[:3])  # 姿态误差
            orn_error_norm = np.linalg.norm(orn_error)  # 姿态误差的范数
            
            # combine position and orientation error  # 将位置误差和姿态误差拼接
            error = np.concatenate([pos_error, orn_error])
            print(f"Iteration {iter}, Position Error: {pos_error_norm:.6f}, Orientation Error: {orn_error_norm:.6f}")  # 打印迭代信息
            
            if pos_error_norm < tolerance and orn_error_norm < tolerance:  # 如果误差小于阈值
                print("IK solved successfully!")  # 打印求解成功
                break
            
            J = self.get_jacobian(current_joints, use_shadow=self.use_shadow_client)  # 计算雅可比矩阵
            
            # damped least squares  # 使用阻尼最小二乘法来求解增量
            delta_q = np.linalg.solve(
                J.T @ J + self.damping * np.eye(self.num_joints),
                J.T @ error
            )
            
            # 更新关节角度，并检查关节限位  # 计算新的关节角度
            new_joints = current_joints + delta_q
            
            # 应用关节限位约束  # 检查并截断超出范围的关节角度
            for i in range(min(len(new_joints), len(joint_limits))):
                lower_limit, upper_limit = joint_limits[i]
                if new_joints[i] < lower_limit:
                    new_joints[i] = lower_limit
                    print(f"警告：关节 {i+1} 超出下限，被截断至 {lower_limit}")
                elif new_joints[i] > upper_limit:
                    new_joints[i] = upper_limit
                    print(f"警告：关节 {i+1} 超出上限，被截断至 {upper_limit}")
            
            # 更新关节角度  # 将新的关节角度设置为当前角度
            current_joints = new_joints
            
            # set joint state in shadow client only during iterations  # 若使用影子客户端，更新影子机器人关节
            if self.use_shadow_client:
                for i, joint_idx in enumerate(self.joint_indices):
                    p.resetJointState(self.shadow_robot_id, joint_idx, current_joints[i], 
                                     physicsClientId=self.shadow_client_id)
            else:
                for i, joint_idx in enumerate(self.joint_indices):
                    p.resetJointState(self.robot_id, joint_idx, current_joints[i])
                
        return current_joints.tolist()  # 返回最终的关节角度解
    
    def __del__(self):  # 析构函数
        """Clean up shadow client if it exists"""  # 这是一段说明文档字符串
        if hasattr(self, 'shadow_client_id') and self.shadow_client_id is not None:  # 如果影子客户端存在
            p.disconnect(physicsClientId=self.shadow_client_id)  # 断开影子客户端连接
