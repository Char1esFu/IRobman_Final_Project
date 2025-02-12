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
        
        # get robot joint index
        self.joint_indices = []
        for i in range(p.getNumJoints(robot_id)):
            joint_info = p.getJointInfo(robot_id, i)
            if joint_info[2] == p.JOINT_REVOLUTE:
                self.joint_indices.append(i)
        self.num_joints = len(self.joint_indices)
        print(f"\nRobot Configuration:")
        print(f"Number of controlled joints: {self.num_joints}")
        print(f"Joint indices: {self.joint_indices}")

    def get_current_ee_pose(self):
        ee_state = p.getLinkState(self.robot_id, self.ee_link_index)
        return np.array(ee_state[0]), np.array(ee_state[1])

    def get_jacobian(self, joint_positions):
        delta = 1e-3  # numerical differentiation step
        jac = np.zeros((6, len(self.joint_indices)))  # 6×n jacobian
        
        # save original position
        original_pos = joint_positions.copy()
        
        current_pos, current_orn = self.get_current_ee_pose()
        
        for i in range(len(self.joint_indices)):
            joint_positions = original_pos.copy()
            joint_positions[i] += delta
            
            # set joint state
            for idx, pos in zip(self.joint_indices, joint_positions):
                p.resetJointState(self.robot_id, idx, pos)
            
            # new position and orientation
            new_pos, new_orn = self.get_current_ee_pose()
            
            # pos jacobian
            jac[:3, i] = (new_pos - current_pos) / delta
            
            # ori jacobian
            # quaternion difference as angular velocity
            orn_diff = p.getDifferenceQuaternion(current_orn.tolist(), new_orn.tolist())
            jac[3:, i] = np.array(orn_diff[:3]) / delta # getQuaternionFromEuler: Convert Euler [roll, pitch, yaw] as in URDF/SDF convention, to quaternion [x,y,z,w]
        
        # reset joint state
        for idx, pos in zip(self.joint_indices, original_pos):
            p.resetJointState(self.robot_id, idx, pos)
            
        return jac

    def solve(self, target_pos, target_orn, current_joint_positions, max_iters=50, tolerance=1e-2):
        """solve IK"""
        current_joints = np.array(current_joint_positions)
        
        for iter in range(max_iters):
            current_pos, current_orn = self.get_current_ee_pose()
            
            pos_error = target_pos - current_pos
            pos_error_norm = np.linalg.norm(pos_error)
            
            orn_error = np.array(p.getDifferenceQuaternion(current_orn.tolist(), target_orn)[:3])
            orn_error_norm = np.linalg.norm(orn_error)
            
            # combine position and orientation error
            error = np.concatenate([pos_error, orn_error])
            print(f"Iteration {iter}, Position Error: {pos_error_norm:.6f}, Orientation Error: {orn_error_norm:.6f}")
            
            if pos_error_norm < tolerance and orn_error_norm < tolerance:
                print("IK solved successfully!")
                break
            
            J = self.get_jacobian(current_joints)
            
            # damped least squares
            delta_q = np.linalg.solve(
                J.T @ J + self.damping * np.eye(self.num_joints),
                J.T @ error
            )
            
            # update joint angles
            current_joints += delta_q
            
            # set joint state
            for i, joint_idx in enumerate(self.joint_indices):
                p.resetJointState(self.robot_id, joint_idx, current_joints[i])
                
        return current_joints.tolist()

def test_ik_solver():
    # init pybullet
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    p.loadURDF("plane.urdf")
    robot_id = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)
    
    ee_link_index = 7
    
    ik_solver = DifferentialIKSolver(robot_id, ee_link_index)
    
    # init joint angles
    initial_joints = [0, -pi/4, 0, -pi/2, 0, pi/3, 0]
    for i, joint_idx in enumerate(ik_solver.joint_indices):
        p.resetJointState(robot_id, joint_idx, initial_joints[i])
    
    # target pose
    target_pos = np.array([0.5, 0.2, 0.7])
    target_orn = p.getQuaternionFromEuler([0, pi/2, 0])
    
    print("\nStarting IK solution...")
    print(f"Target position: {target_pos}")
    print(f"Target orientation: {target_orn}")
    
    # solve IK
    try:
        new_joints = ik_solver.solve(target_pos, target_orn, initial_joints)
        print("\nFinal joint angles:", new_joints)
        
        # apply solution
        for i, joint_idx in enumerate(ik_solver.joint_indices):
            p.setJointMotorControl2(robot_id, joint_idx, p.POSITION_CONTROL, new_joints[i])
        
        # keep simulation running
        for _ in range(1000):
            p.stepSimulation()
            time.sleep(1./240.)
            
    except Exception as e:
        print(f"Error during IK solution: {str(e)}")
    finally:
        p.disconnect()

if __name__ == "__main__":
    test_ik_solver()