import pybullet as p
import pybullet_data
import numpy as np
from math import pi
import time

class DifferentialIKSolver:
    def __init__(self, robot_id, ee_link_index, damping=0.001, use_shadow_client=True, damping_max=0.1):
        self.robot_id = robot_id
        self.ee_link_index = ee_link_index
        self.damping = damping
        self.damping_max = damping_max  # 最大阻尼系数
        self.use_shadow_client = use_shadow_client
        
        # 奇异点处理参数
        self.sv_threshold = 0.05  # 奇异值阈值，低于此值认为接近奇异点
        self.cn_threshold = 100.0  # 条件数阈值，高于此值认为接近奇异点
        self.adaptive_damping = True  # 是否使用自适应阻尼
        
        # 在接近关节限制时的斥力控制
        self.use_joint_limits_avoidance = True
        self.joint_limit_threshold = 0.2  # 关节角度接近限制的阈值（弧度），增加到0.2
        # 为特定关节设置单独的阈值
        self.joint_specific_thresholds = {}
        
        # get robot joint index
        self.joint_indices = []
        self.joint_limits_lower = []
        self.joint_limits_upper = []
        
        for i in range(p.getNumJoints(robot_id)):
            joint_info = p.getJointInfo(robot_id, i)
            if joint_info[2] == p.JOINT_REVOLUTE:
                joint_idx = len(self.joint_indices)
                self.joint_indices.append(i)
                self.joint_limits_lower.append(joint_info[8])  # 关节下限
                self.joint_limits_upper.append(joint_info[9])  # 关节上限
                
                # 为第4关节（panda_joint4）设置更大的阈值
                if joint_info[1].decode() == "panda_joint4":
                    self.joint_specific_thresholds[joint_idx] = 0.4  # 为第4关节设置更大的阈值
        
        self.num_joints = len(self.joint_indices)
        
        # Create shadow client for IK calculations if needed
        if self.use_shadow_client:
            # Create a DIRECT physics client for shadow calculations
            self.shadow_client_id = p.connect(p.DIRECT)
            
            # Get the real robot's base position and orientation
            robot_base_pos, robot_base_orn = p.getBasePositionAndOrientation(self.robot_id)
            
            # Load the same robot in the shadow client with the same position and orientation
            p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.shadow_client_id)
            self.shadow_robot_id = p.loadURDF("franka_panda/panda.urdf", 
                                             basePosition=robot_base_pos,
                                             baseOrientation=robot_base_orn,
                                             useFixedBase=True, 
                                             physicsClientId=self.shadow_client_id)
            
            # Copy the current joint states from the real robot to the shadow robot
            for joint_idx in self.joint_indices:
                joint_state = p.getJointState(self.robot_id, joint_idx)
                joint_pos = joint_state[0]  # Current position
                joint_vel = 0.0  # Reset velocity to zero
                p.resetJointState(self.shadow_robot_id, joint_idx, 
                                 targetValue=joint_pos,
                                 targetVelocity=joint_vel,
                                 physicsClientId=self.shadow_client_id)
        else:
            self.shadow_client_id = None
            self.shadow_robot_id = None
        
        print(f"\n机器人配置:")
        print(f"控制关节数量: {self.num_joints}")
        print(f"关节索引: {self.joint_indices}")
        
        if self.use_shadow_client:
            print(f"Shadow客户端ID: {self.shadow_client_id}")
            print(f"Shadow机器人ID: {self.shadow_robot_id}")

    def get_current_ee_pose(self, use_shadow=False):
        """Get the current end effector pose"""
        if use_shadow and self.use_shadow_client:
            client_id = self.shadow_client_id
            robot_id = self.shadow_robot_id
        else:
            # Use default client ID (no need to specify)
            client_id = 0  # Default client ID
            robot_id = self.robot_id
        
        # Switch to appropriate client
        p.setPhysicsEngineParameter(enableConeFriction=0, physicsClientId=client_id)
        
        ee_state = p.getLinkState(robot_id, self.ee_link_index, physicsClientId=client_id)
        return np.array(ee_state[0]), np.array(ee_state[1])

    def get_jacobian(self, joint_positions, use_shadow=True):
        """Calculate the Jacobian matrix"""
        delta = 1e-3  # numerical differentiation step
        jac = np.zeros((6, len(self.joint_indices)))  # 6×n jacobian
        
        if use_shadow and self.use_shadow_client:
            client_id = self.shadow_client_id
            robot_id = self.shadow_robot_id
        else:
            client_id = 0  # Default client ID
            robot_id = self.robot_id
        
        # Switch to appropriate client
        p.setPhysicsEngineParameter(enableConeFriction=0, physicsClientId=client_id)
        
        # save original position
        original_pos = joint_positions.copy()
        
        # Set initial joint positions
        for idx, pos in zip(self.joint_indices, original_pos):
            p.resetJointState(robot_id, idx, pos, physicsClientId=client_id)
            
        current_pos, current_orn = self.get_current_ee_pose(use_shadow=use_shadow)
        
        for i in range(len(self.joint_indices)):
            joint_positions = original_pos.copy()
            joint_positions[i] += delta
            
            # set joint state
            for idx, pos in zip(self.joint_indices, joint_positions):
                p.resetJointState(robot_id, idx, pos, physicsClientId=client_id)
            
            # new position and orientation
            new_pos, new_orn = self.get_current_ee_pose(use_shadow=use_shadow)
            
            # pos jacobian
            jac[:3, i] = (new_pos - current_pos) / delta
            
            # ori jacobian
            # quaternion difference as angular velocity
            orn_diff = p.getDifferenceQuaternion(current_orn.tolist(), new_orn.tolist(), physicsClientId=client_id)
            jac[3:, i] = np.array(orn_diff[:3]) / delta
        
        # reset joint state
        for idx, pos in zip(self.joint_indices, original_pos):
            p.resetJointState(robot_id, idx, pos, physicsClientId=client_id)
            
        return jac

    def calculate_manipulability(self, jacobian):
        """计算机械臂的可操作度（操作度）
        
        可操作度是衡量机械臂距离奇异点远近的指标
        """
        # 计算雅可比矩阵的奇异值
        svd_values = np.linalg.svd(jacobian, compute_uv=False)
        
        # 最小奇异值表示距离奇异点的接近程度
        min_sv = np.min(svd_values)
        
        # 条件数 = 最大奇异值/最小奇异值
        # 条件数越大，表示越接近奇异点
        condition_number = np.max(svd_values) / (min_sv + 1e-10)
        
        # 计算可操作度（奇异值的乘积）
        manipulability = np.prod(svd_values)
        
        return min_sv, condition_number, manipulability

    def calculate_dynamic_damping(self, min_singular_value, condition_number):
        """根据最小奇异值和条件数动态调整阻尼参数"""
        # 如果不使用自适应阻尼，直接返回基础阻尼值
        if not self.adaptive_damping:
            return self.damping, min_singular_value < self.sv_threshold or condition_number > self.cn_threshold
            
        if min_singular_value < self.sv_threshold or condition_number > self.cn_threshold:
            # 接近奇异点时，增大阻尼
            # 根据最小奇异值动态调整阻尼系数
            damping_factor = (1.0 - min_singular_value/self.sv_threshold) if min_singular_value < self.sv_threshold else 0
            damping_factor = max(damping_factor, (condition_number - self.cn_threshold)/self.cn_threshold if condition_number > self.cn_threshold else 0)
            
            # 限制阻尼系数范围
            damping_factor = np.clip(damping_factor, 0.0, 1.0)
            
            # 计算动态阻尼值 - 使用指数增长而不是线性增长
            # 当非常接近奇异点时，阻尼系数快速增大
            dynamic_damping = self.damping + damping_factor**0.5 * (self.damping_max - self.damping)
            
            return dynamic_damping, True
        else:
            return self.damping, False
    
    def calculate_joint_limit_avoidance(self, joint_positions):
        """计算关节限制避免的梯度"""
        if not self.use_joint_limits_avoidance:
            return np.zeros(self.num_joints)
            
        gradient = np.zeros(self.num_joints)
        
        for i in range(self.num_joints):
            lower = self.joint_limits_lower[i]
            upper = self.joint_limits_upper[i]
            pos = joint_positions[i]
            
            # 使用特定关节的阈值或默认阈值
            threshold = self.joint_specific_thresholds.get(i, self.joint_limit_threshold)
            
            # 检查是否接近关节限制
            if pos < lower + threshold:
                # 接近下限，添加正梯度
                distance = pos - lower
                factor = 1.0 - distance / threshold
                # 对于第4关节（索引3）使用更大的梯度强度
                gradient_strength = 0.3 if i == 3 else 0.1
                gradient[i] = factor * gradient_strength
            elif pos > upper - threshold:
                # 接近上限，添加负梯度
                distance = upper - pos
                factor = 1.0 - distance / threshold
                # 对于第4关节（索引3）使用更大的梯度强度
                gradient_strength = 0.3 if i == 3 else 0.1
                gradient[i] = -factor * gradient_strength
                
        return gradient

    def solve(self, target_pos, target_orn, current_joint_positions, max_iters=50, tolerance=1e-3):
        """solve IK using shadow client if available"""
        current_joints = np.array(current_joint_positions)
        
        # If using shadow client, set initial joint positions in shadow robot
        if self.use_shadow_client:
            for i, joint_idx in enumerate(self.joint_indices):
                p.resetJointState(self.shadow_robot_id, joint_idx, current_joints[i], 
                                 physicsClientId=self.shadow_client_id)
        
        # 记录收敛历史
        position_errors = []
        orientation_errors = []
        min_singular_values = []
        condition_numbers = []
        damping_values = []
        
        for iter in range(max_iters):
            # Use shadow client for calculations if available
            current_pos, current_orn = self.get_current_ee_pose(use_shadow=self.use_shadow_client)
            
            pos_error = target_pos - current_pos
            pos_error_norm = np.linalg.norm(pos_error)
            
            client_id = self.shadow_client_id if self.use_shadow_client else 0  # Default client ID
            orn_error = np.array(p.getDifferenceQuaternion(current_orn.tolist(), target_orn, 
                                                          physicsClientId=client_id)[:3])
            orn_error_norm = np.linalg.norm(orn_error)
            
            # combine position and orientation error
            error = np.concatenate([pos_error, orn_error])
            
            # 记录收敛历史
            position_errors.append(pos_error_norm)
            orientation_errors.append(orn_error_norm)
            
            if pos_error_norm < tolerance and orn_error_norm < tolerance:
                print("逆运动学求解成功!")
                break
            
            J = self.get_jacobian(current_joints, use_shadow=self.use_shadow_client)
            
            # 计算奇异值和可操作度信息
            min_sv, condition_number, manipulability = self.calculate_manipulability(J)
            min_singular_values.append(min_sv)
            condition_numbers.append(condition_number)
            
            # 动态调整阻尼系数
            current_damping, near_singular = self.calculate_dynamic_damping(min_sv, condition_number)
            damping_values.append(current_damping)
            
            # 计算关节限制避免的梯度
            joint_limit_gradient = self.calculate_joint_limit_avoidance(current_joints)
            
            # 打印详细信息
            singular_status = "警告：接近奇异点！" if near_singular else "状态正常"
            print(f"迭代 {iter}, 位置误差: {pos_error_norm:.6f}, 姿态误差: {orn_error_norm:.6f}")
            print(f"奇异值分析 - 最小奇异值: {min_sv:.6f}, 条件数: {condition_number:.2f}, 可操作度: {manipulability:.6e}")
            print(f"阻尼系数: {current_damping:.6f}, {singular_status}")
            
            # 打印关节限位信息
            for i in range(self.num_joints):
                joint_name = p.getJointInfo(self.robot_id, self.joint_indices[i])[1].decode()
                lower = self.joint_limits_lower[i]
                upper = self.joint_limits_upper[i]
                current = current_joints[i]
                
                # 计算到限位的距离百分比
                lower_percent = (current - lower) / (upper - lower) * 100
                upper_percent = (upper - current) / (upper - lower) * 100
                
                # 只打印接近限位的关节
                threshold = self.joint_specific_thresholds.get(i, self.joint_limit_threshold)
                if current < lower + threshold or current > upper - threshold:
                    print(f"关节 {joint_name}: 当前值={current:.4f}, 下限={lower:.4f}, 上限={upper:.4f}")
                    print(f"  距下限: {lower_percent:.1f}%, 距上限: {upper_percent:.1f}%")
            
            # 如果极度接近奇异点，尝试退出当前求解
            if min_sv < 0.01 and iter > 5:
                print("警告：极度接近奇异点，求解可能不稳定！")
                # 可以在这里添加特殊处理逻辑
            
            # damped least squares with dynamic damping
            try:
                delta_q = np.linalg.solve(
                    J.T @ J + current_damping * np.eye(self.num_joints),
                    J.T @ error
                )
                
                # 添加关节限制避免的梯度
                delta_q += joint_limit_gradient
                
                # 限制步长，防止大步伐导致的不稳定
                step_size = np.linalg.norm(delta_q)
                if step_size > 0.2:  # 最大步长限制
                    delta_q = delta_q * 0.2 / step_size
                
                # update joint angles
                current_joints += delta_q
                
                # 确保关节角度在限制范围内
                for i in range(self.num_joints):
                    current_joints[i] = np.clip(current_joints[i], 
                                              self.joint_limits_lower[i], 
                                              self.joint_limits_upper[i])
                    
                    # 对第4关节（索引3）进行特殊处理，确保它远离上限
                    if i == 3:  # panda_joint4
                        # 如果太接近上限（0.0），强制向下移动
                        if current_joints[i] > -0.1:  # 如果接近0
                            current_joints[i] = -0.1  # 强制设置为一个更小的值
                
            except np.linalg.LinAlgError:
                print("警告：线性代数错误，可能是雅可比矩阵接近奇异！")
                # 增加阻尼并重试
                delta_q = np.linalg.solve(
                    J.T @ J + self.damping_max * np.eye(self.num_joints),
                    J.T @ error
                )
                # 使用更小的步长
                current_joints += delta_q * 0.1
            
            # set joint state in shadow client only during iterations
            if self.use_shadow_client:
                for i, joint_idx in enumerate(self.joint_indices):
                    p.resetJointState(self.shadow_robot_id, joint_idx, current_joints[i], 
                                     physicsClientId=self.shadow_client_id)
            else:
                for i, joint_idx in enumerate(self.joint_indices):
                    p.resetJointState(self.robot_id, joint_idx, current_joints[i])
        
        # 检查解算是否成功
        if iter >= max_iters - 1:
            print("警告：达到最大迭代次数，求解可能不完全收敛")
            
        # 分析收敛历史和奇异性
        if len(min_singular_values) > 0:
            lowest_sv = min(min_singular_values)
            highest_cn = max(condition_numbers)
            print(f"求解过程分析 - 最低奇异值: {lowest_sv:.6f}, 最高条件数: {highest_cn:.2f}")
            print(f"使用的阻尼范围: {min(damping_values):.6f} 到 {max(damping_values):.6f}")
            
        return current_joints.tolist()
    
    def __del__(self):
        """Clean up shadow client if it exists"""
        if hasattr(self, 'shadow_client_id') and self.shadow_client_id is not None:
            p.disconnect(physicsClientId=self.shadow_client_id)