import pybullet as p
import pybullet_data
import numpy as np
from math import pi
import time
from src.ik_solver import DifferentialIKSolver

def run_ik_test(ik_solver, target_pos, target_orn, initial_joints):
    """single IK test"""
    print("\nTesting IK solution...")
    print(f"Target position: {target_pos}")
    print(f"Target orientation: {target_orn}")
    
    try:
        new_joints = ik_solver.solve(target_pos, target_orn, initial_joints)
        print("\nSolution found:")
        print("Final joint angles:", new_joints)
        
        # validate solution
        for i, joint_idx in enumerate(ik_solver.joint_indices):
            p.resetJointState(ik_solver.robot_id, joint_idx, new_joints[i])
        p.stepSimulation()
        
        # acquire final end-effector pose
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
    """generate random target pose"""
    # position limits
    x = np.random.uniform(0.3, 0.4)    
    y = np.random.uniform(-0.3, 0.3)   
    z = np.random.uniform(0.3, 0.4)  
    
    # orientation limits
    roll = np.random.uniform(-pi/4, pi/4)
    pitch = np.random.uniform(-pi/4, pi/4)
    yaw = np.random.uniform(-pi/2, pi/2)
    
    return np.array([x, y, z]), p.getQuaternionFromEuler([roll, pitch, yaw])

def generate_initial_joints(num_joints):
    """generate random initial joint angles"""
    joint_limits = [
        (-2.8973, 2.8973),    # joint 1
        (-1.7628, 1.7628),    # joint 2
        (-2.8973, 2.8973),    # joint 3
        (-3.0718, -0.0698),   # joint 4
        (-2.8973, 2.8973),    # joint 5
        (-0.0175, 3.7525),    # joint 6
        (-2.8973, 2.8973)     # joint 7
    ]
    
    return [np.random.uniform(low, high) for low, high in joint_limits[:num_joints]]

def test_ik_solver(num_tests=50):
    """random IK solver test"""
    # init pybullet
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    p.loadURDF("plane.urdf")
    robot_id = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)
    
    ee_link_index = 7
    
    ik_solver = DifferentialIKSolver(robot_id, ee_link_index)
    
    results = []
    successful_tests = 0
    
    for i in range(num_tests):
        print(f"\n=== Random Test {i+1}/{num_tests} ===")
        
        # generate random target pose and initial joint angles
        target_pos, target_orn = generate_random_pose()
        initial_joints = generate_initial_joints(len(ik_solver.joint_indices))
        
        # set initial joint angles
        for j, joint_idx in enumerate(ik_solver.joint_indices):
            p.resetJointState(ik_solver.robot_id, joint_idx, initial_joints[j])
        p.stepSimulation()
        
        # run IK test
        solution, pos_error, orn_error = run_ik_test(
            ik_solver,
            target_pos,
            target_orn,
            initial_joints
        )
        
        # record test results
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
        
        # pause for observation
        time.sleep(0.5)
    
    print("\n=== Test Summary ===")
    print(f"Total tests: {num_tests}")
    print(f"Successful tests: {successful_tests}")
    print(f"Success rate: {(successful_tests/num_tests)*100:.2f}%")
    
    # print error statistics for successful tests
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