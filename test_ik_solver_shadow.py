import pybullet as p
import pybullet_data
import numpy as np
from math import pi
import time
from src.ik_solver_shadow import DifferentialIKSolver  # 新版本，含影子客户端和抓取关节支持

def run_ik_test(ik_solver, target_pos, target_orn, initial_joints):
    """对单个 IK 测试案例进行求解，并返回解与误差"""
    print("\nTesting IK solution...")
    print(f"Target position: {target_pos}")
    print(f"Target orientation: {target_orn}")
    
    try:
        new_joints = ik_solver.solve(target_pos, target_orn, initial_joints)
        print("\nSolution found:")
        print("Final joint angles:", new_joints)
        
        # 验证解（注意，这里操作的是影子客户端中的机器人）
        for idx, angle in zip(ik_solver.joint_indices, new_joints):
            p.resetJointState(ik_solver.robot_id, idx, angle, physicsClientId=ik_solver.cid)
        p.stepSimulation(physicsClientId=ik_solver.cid)
        
        # 获取末端执行器实际位姿
        final_pos, final_orn = ik_solver.get_current_ee_pose()
        pos_error = np.linalg.norm(target_pos - final_pos)
        orn_error = np.linalg.norm(np.array(p.getDifferenceQuaternion(final_orn.tolist(), target_orn)[:3]))
        
        print("Final errors:")
        print(f"Position error: {pos_error:.6f}")
        print(f"Orientation error: {orn_error:.6f}")
        
        return new_joints, pos_error, orn_error
    except Exception as e:
        print(f"Error during IK solution: {str(e)}")
        return None, None, None

def generate_random_pose():
    """生成随机目标末端位姿"""
    x = np.random.uniform(0.3, 0.4)
    y = np.random.uniform(-0.3, 0.3)
    z = np.random.uniform(0.3, 0.4)
    roll = np.random.uniform(-pi/4, pi/4)
    pitch = np.random.uniform(-pi/4, pi/4)
    yaw = np.random.uniform(-pi/2, pi/2)
    return np.array([x, y, z]), p.getQuaternionFromEuler([roll, pitch, yaw])

def generate_initial_joints(num_joints):
    """生成随机初始关节角度，针对新版 IK 求解器（num_joints=9）"""
    if num_joints == 9:
        # 对于 Panda 机器人，假设前7个关节为机械臂，后2个为抓取关节
        joint_limits = [
            (-2.9671, 2.9671),    # joint 1
            (-1.8326, 1.8326),    # joint 2
            (-2.9671, 2.9671),    # joint 3
            (-3.1416, 0.0),       # joint 4
            (-2.9671, 2.9671),    # joint 5
            (-0.0873, 3.8223),    # joint 6
            (-2.9671, 2.9671),    # joint 7
            (0.0, 0.04),          # finger joint 1
            (0.0, 0.04)           # finger joint 2
        ]
    else:
        # 如果不是 9，自行填充默认限制
        joint_limits = [(-2.9671, 2.9671)] * num_joints
    return [np.random.uniform(low, high) for low, high in joint_limits[:num_joints]]

def test_ik_solver(num_tests=50):
    """批量测试新版 IK 求解器（影子客户端 + 考虑抓取关节）"""
    # 初始化主仿真（用于显示，但实际 IK 计算在影子客户端中进行）
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    # 加载主仿真中的机器人（仅作显示用途）
    robot_id = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)
    
    # 对于包含抓取关节的情况，末端 link 索引设为 11
    ee_link_index = 11
    
    # 创建 IK 求解器实例，开启影子客户端（注意：影子客户端中的机器人与主仿真无关）
    ik_solver = DifferentialIKSolver(
        robot_id=robot_id,            # 仅作参数参考
        ee_link_index=ee_link_index,
        damping=0.05,
        use_shadow=True,              # 开启影子模式
        urdf_path="franka_panda/panda.urdf",
        base_position=[0, 0, 0],
        use_fixed_base=True
    )
    
    results = []
    successful_tests = 0
    
    for i in range(num_tests):
        print(f"\n=== Random Test {i+1}/{num_tests} ===")
        
        # 生成随机目标末端位姿
        target_pos, target_orn = generate_random_pose()
        # 根据 IK 求解器采集的关节数（应为 9）生成初始角度
        initial_joints = generate_initial_joints(len(ik_solver.joint_indices))
        
        # 在影子客户端中设置初始关节状态
        for j, joint_idx in enumerate(ik_solver.joint_indices):
            p.resetJointState(ik_solver.robot_id, joint_idx, initial_joints[j], physicsClientId=ik_solver.cid)
        p.stepSimulation(physicsClientId=ik_solver.cid)
        
        # 运行单个 IK 测试
        solution, pos_error, orn_error = run_ik_test(ik_solver, target_pos, target_orn, initial_joints)
        
        success = solution is not None and pos_error < 0.01 and orn_error < 0.1
        if success:
            successful_tests += 1
            
        results.append({
            'test_id': i+1,
            'target_pos': target_pos,
            'target_orn': target_orn,
            'success': success,
            'pos_error': pos_error,
            'orn_error': orn_error
        })
        
        time.sleep(0.5)
    
    print("\n=== Test Summary ===")
    print(f"Total tests: {num_tests}")
    print(f"Successful tests: {successful_tests}")
    print(f"Success rate: {(successful_tests/num_tests)*100:.2f}%")
    
    successful_results = [r for r in results if r['success']]
    if successful_results:
        pos_errors = [r['pos_error'] for r in successful_results]
        orn_errors = [r['orn_error'] for r in successful_results]
        print("\nError Statistics for Successful Tests:")
        print(f"Position Error - Mean: {np.mean(pos_errors):.6f}, Max: {np.max(pos_errors):.6f}")
        print(f"Orientation Error - Mean: {np.mean(orn_errors):.6f}, Max: {np.max(orn_errors):.6f}")
    
    time.sleep(5)
    p.disconnect()

if __name__ == "__main__":
    test_ik_solver(50)
