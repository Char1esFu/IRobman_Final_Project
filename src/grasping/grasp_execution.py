import cv2
import numpy as np
import pybullet as p
import time

from typing import Any, Dict

from src.grasping.grasp_generation import GraspGeneration
from src.ik_solver.ik_solver import DifferentialIKSolver
from src.obstacle_tracker.obstacle_tracker import ObstacleTracker
from src.path_planning.simple_planning import SimpleTrajectoryPlanner



class GraspExecution:
    """Robot grasping execution class, responsible for planning and executing complete grasping actions"""
    
    def __init__(self, sim, config: Dict[str, Any], bbox_center, bbox_rotation_matrix):
        """
        Initialize grasping executor
        
        Parameters:
            sim: Simulation environment object
        """
        self.sim = sim
        self.ik_solver = DifferentialIKSolver(sim.robot.id, sim.robot.ee_idx, damping=0.05)
        self.trajectory_planner = SimpleTrajectoryPlanner
        self.config = config
        self.bbox_center = bbox_center
        self.bbox_rotation_matrix = bbox_rotation_matrix
    
    def execute_grasp(self, pose1_pos, pose1_orn, pose2_pos, pose2_orn):
        """
        Execute complete grasping process
        
        参数：
            best_grasp: 最佳抓取姿态 (R, grasp_center)
            grasp_poses: 可选的预计算姿态 (pose1_pos, pose1_orn, pose2_pos, pose2_orn)
            
        Returns:
            bool: True if grasping is successful, False otherwise
        """
        
        # 获取当前机器人关节角度
        start_joints = self.sim.robot.get_joint_positions()
        
        # Solve IK for pre-grasp position
        target_joints = self.ik_solver.solve(pose1_pos, pose1_orn, start_joints, max_iters=50, tolerance=0.001)
        
        if target_joints is None:
            print("IK cannot be solved, cannot move to pre-grasp position")
            return False
        
        # Generate and execute trajectory to pre-grasp position
        trajectory = self.trajectory_planner.generate_joint_trajectory(start_joints, target_joints, steps=100)
        self._execute_trajectory(trajectory)
        
        # Open gripper
        self.open_gripper()
        
        # Move to final grasp position
        current_joints = self.sim.robot.get_joint_positions()
        pose2_trajectory = self.trajectory_planner.generate_cartesian_trajectory(
            self.sim.robot.id, 
            self.sim.robot.arm_idx, 
            self.sim.robot.ee_idx,
            current_joints, 
            pose2_pos, 
            pose2_orn, 
            steps=50
        )
        
        if not pose2_trajectory:
            print("Cannot generate trajectory to final grasp position")
            return False
        
        self._execute_trajectory(pose2_trajectory)
        
        # Wait for stabilization
        self._wait(0.5)
        
        # Close gripper to grasp object
        # self.close_gripper()
        self.close_gripper_hybrid()
        
    def _execute_trajectory(self, trajectory, speed=1/240.0):
        """Execute trajectory"""
        for joint_target in trajectory:
            self.sim.robot.position_control(joint_target)
            for _ in range(1):
                self.sim.step()
                time.sleep(speed)
    
    def _wait(self, seconds):
        """Wait for specified seconds"""
        steps = int(seconds * 240)
        for _ in range(steps):
            self.sim.step()
            time.sleep(1/240.)
    
    def open_gripper(self, width=0.04):
        """Open robot gripper"""
        p.setJointMotorControlArray(
            self.sim.robot.id,
            jointIndices=self.sim.robot.gripper_idx,
            controlMode=p.POSITION_CONTROL,
            targetPositions=[width, width]
        )
        self._wait(0.5)
    
    def close_gripper(self, width=0.01):
        """Close robot gripper to grasp object"""
        p.setJointMotorControlArray(
            self.sim.robot.id,
            jointIndices=self.sim.robot.gripper_idx,
            controlMode=p.POSITION_CONTROL,
            targetPositions=[width, width]
        )
        self._wait(1.0)
    
    def close_gripper_hybrid(self, target_width=0.01, max_force=10.0):
        """混合位置和力控制来闭合爪子"""
        p.setJointMotorControlArray(
            self.sim.robot.id,
            jointIndices=self.sim.robot.gripper_idx,
            controlMode=p.POSITION_CONTROL,
            targetPositions=[target_width, target_width],
            forces=[max_force, max_force]
        )
        self._wait(1.0)
    
    def lift_object(self, height=0.5):
        """Grasp object and lift it to specified height"""
        # Get current end-effector position and orientation
        current_ee_pos, current_ee_orn = self.sim.robot.get_ee_pose()
        
        # Calculate lifted position
        lift_pos = current_ee_pos.copy()
        lift_pos[2] += height
        
        # Get current joint angles
        current_joints = self.sim.robot.get_joint_positions()
        
        # Solve IK for lifted position
        lift_target_joints = self.ik_solver.solve(lift_pos, current_ee_orn, current_joints, max_iters=50, tolerance=0.001)
        
        if lift_target_joints is None:
            print("IK cannot be solved for lifted position, cannot lift object")
            return False
        
        # Generate and execute lifting trajectory
        lift_trajectory = self.trajectory_planner.generate_joint_trajectory(current_joints, lift_target_joints, steps=100)
        
        if not lift_trajectory:
            print("Cannot generate lifting trajectory")
            return False
        
        self._execute_trajectory(lift_trajectory, speed=1/240.0)
        return True

    def execute_complete_grasp(self, point_clouds, visualize=True):
        """
        Execute complete process of grasping planning and execution
        
        Parameters:
        point_clouds: Collected point cloud data
        visualize: Whether to visualize grasping process
        
        Returns:
        success: True if grasping is successful, False otherwise
        self: Grasping executor object (if grasping is successful)
        """        
        grasp_generator = GraspGeneration(self.bbox_center, self.bbox_rotation_matrix, self.sim)
        pose1_pos, pose1_orn, pose2_pos, pose2_orn = grasp_generator.final_compute_poses(point_clouds, visualize)
        # Execute grasping (pass calculated pose)
        print("\nStarting to execute grasping...")
        self.execute_grasp(pose1_pos, pose1_orn, pose2_pos, pose2_orn)

        lift_success = self.lift_object()

        is_success = self.is_grasped(mask_id=5, distance_threshold=0.15)

        if is_success and lift_success:
            print("\nGrasping successful!")
        else:
            print("\nGrasping failed...")
            return False, False
        
        return True, True

    def is_grasped(self, mask_id=5, distance_threshold=0.15):
        """
        判断物体是否已被成功抓取
        
        通过计算物体可见部分的中心与末端执行器之间的距离来判断
        
        参数:
            mask_id: 被抓取物体的掩码ID，默认为5
            distance_threshold: 距离阈值，小于此值视为抓取成功，默认为0.1
            
        返回:
            bool: 如果物体被成功抓取返回True，否则返回False
        """
        # 获取末端执行器相机图像
        rgb, depth, seg = self.sim.get_static_renders()
        
        # 创建ObstacleTracker实例
        tracker = ObstacleTracker(n_obstacles=2, exp_settings= self.config)
        
        # 检查物体ID是否在分割掩码中
        if mask_id not in np.unique(seg):
            print(f"警告: 物体ID {mask_id} 不在当前分割掩码中")
            print(f"分割掩码中的可用ID: {np.unique(seg)}")
            return False
        
        # 创建物体掩码
        mask = (seg == mask_id).astype(np.uint8)
        
        # 找到轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print(f"警告: 物体ID {mask_id} 没有找到有效轮廓")
            return False
        
        # 使用最大的轮廓
        contour = max(contours, key=cv2.contourArea)

        # 计算轮廓中心
        M = cv2.moments(contour)
        if M['m00'] == 0:
            print(f"警告: 物体ID {mask_id} 的轮廓面积为零")
            return False
        
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        
        # 检查深度
        depth_buffer = depth[cy, cx]
        metric_depth = tracker.convert_depth_to_meters(depth_buffer)
        
        # 使用ObstacleTracker的方法将像素坐标转换为世界坐标
        # 不考虑半径偏移，因为我们只关心表面点
        object_pos = tracker.pixel_to_world(cx, cy, metric_depth)
        
        # 获取末端执行器位置
        ee_pos, _ = self.sim.robot.get_ee_pose()
        
        # 计算末端执行器与物体中心的距离
        distance = np.linalg.norm(np.array(ee_pos) - object_pos)
        
        # 判断距离是否小于阈值
        is_success = distance < distance_threshold
        
        # 输出调试信息
        print(f"物体中心世界坐标: {object_pos}")
        print(f"末端执行器坐标: {ee_pos}")
        print(f"距离: {distance}, 阈值: {distance_threshold}")
        print(f"抓取{'成功' if is_success else '失败'}")
        
        return is_success 