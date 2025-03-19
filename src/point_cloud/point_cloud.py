import os
import glob
import numpy as np
import pybullet as p
import open3d as o3d
import time
import random
from scipy.spatial.transform import Rotation
from src.path_planning.simple_planning import SimpleTrajectoryPlanner

class PointCloudCollector:
    def __init__(self, config, sim):
        """
        Initialize point cloud collector
        
        Parameters:
        config: Configuration dictionary
        sim: Simulation environment object
        """
        self.config = config
        self.sim = sim
        
    def _convert_depth_to_meters(self, depth_buffer, near, far):
        """
        Convert depth buffer values to actual distances (meters)
        
        Parameters:
        depth_buffer: Depth buffer values from PyBullet
        near, far: Near/far plane distances
        
        Returns:
        Actual depth values in meters
        """
        return far * near / (far - (far - near) * depth_buffer)

    def _get_camera_intrinsic(self, width, height, fov):
        """
        Calculate intrinsic matrix from camera parameters
        
        Parameters:
        width: Image width (pixels)
        height: Image height (pixels)
        fov: Vertical field of view (degrees)

        Returns:
        Camera intrinsic matrix
        """    
        # Calculate focal length
        f = height / (2 * np.tan(np.radians(fov / 2)))
        
        # Calculate principal point
        cx = width / 2
        cy = height / 2
        
        intrinsic_matrix = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1]
        ])
        
        return intrinsic_matrix

    def _depth_image_to_point_cloud(self, depth_image, mask, rgb_image, intrinsic_matrix):
        """
        Convert depth image to point cloud in camera coordinate system
        
        Parameters:
        depth_image: Depth image (meters)
        mask: Target object mask (boolean array)
        rgb_image: RGB image
        intrinsic_matrix: Camera intrinsic matrix
        
        Returns:
        Point cloud in camera coordinates (N,3) and corresponding colors (N,3)
        """
        # Extract pixel coordinates from target mask
        rows, cols = np.where(mask)
        
        if len(rows) == 0:
            raise ValueError("No valid pixels found in target mask")
        
        # Extract depth values for these pixels
        depths = depth_image[rows, cols]
        
        # Convert image coordinates to camera coordinates
        fx = intrinsic_matrix[0, 0]
        fy = intrinsic_matrix[1, 1]
        cx = intrinsic_matrix[0, 2]
        cy = intrinsic_matrix[1, 2]
        
        # Calculate camera coordinates
        x = -(cols - cx) * depths / fx # Negative sign due to PyBullet camera direction
        y = -(rows - cy) * depths / fy
        z = depths
        
        # Stack points
        points = np.vstack((x, y, z)).T
        
        # Extract RGB colors
        colors = rgb_image[rows, cols, :3].astype(np.float64) / 255.0
        
        return points, colors

    def _transform_points_to_world(self, points, camera_extrinsic):
        """
        Transform points from camera coordinate system to world coordinate system
        
        Parameters:
        points: Point cloud in camera coordinates (N,3)
        camera_extrinsic: Camera extrinsic matrix (4x4)
        
        Returns:
        Point cloud in world coordinates (N,3)
        """
        # Convert point cloud to homogeneous coordinates
        points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
        
        # Transform point cloud using extrinsic matrix
        world_points_homogeneous = (camera_extrinsic @ points_homogeneous.T).T # Points in rows
        
        # Convert back to non-homogeneous coordinates
        world_points = world_points_homogeneous[:, :3]
        
        return world_points

    def _get_camera_extrinsic(self, camera_pos, camera_R):
        """
        Build camera extrinsic matrix (transformation from camera to world coordinates)
        
        Parameters:
        camera_pos: Camera position in world coordinates
        camera_R: Camera rotation matrix (3x3)
        
        Returns:
        Camera extrinsic matrix (4x4)
        """
        # Build 4x4 extrinsic matrix
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = camera_R
        extrinsic[:3, 3] = camera_pos
        
        return extrinsic

    def _build_object_point_cloud_ee(self, rgb, depth, seg, target_mask_id, camera_pos, camera_R):
        """
        Build object point cloud using RGB, depth, segmentation data from end effector camera
        
        Parameters:
        rgb: RGB image
        depth: Depth buffer values
        seg: Segmentation mask
        target_mask_id: Target object ID
        camera_pos: Camera position in world coordinates
        camera_R: Camera rotation matrix (from camera to world coordinates)
        
        Returns:
        Open3D point cloud object
        """
        # Read camera parameters
        cam_cfg = self.config["world_settings"]["camera"]
        width = cam_cfg["width"]
        height = cam_cfg["height"]
        fov = cam_cfg["fov"]  # Vertical FOV
        near = cam_cfg["near"]
        far = cam_cfg["far"]
        
        # Create target object mask
        object_mask = (seg == target_mask_id)
        if np.count_nonzero(object_mask) == 0:
            raise ValueError(f"Target mask ID {target_mask_id} not found in segmentation")
        
        # Extract target object depth buffer values
        metric_depth = self._convert_depth_to_meters(depth, near, far)
        
        # Get intrinsic matrix
        intrinsic_matrix = self._get_camera_intrinsic(width, height, fov)
        
        # Convert depth image to point cloud
        points_cam, colors = self._depth_image_to_point_cloud(metric_depth, object_mask, rgb, intrinsic_matrix)
        
        # Build camera extrinsic matrix
        camera_extrinsic = self._get_camera_extrinsic(camera_pos, camera_R)
        
        # Transform points to world coordinate system
        points_world = self._transform_points_to_world(points_cam, camera_extrinsic)
        
        # Create Open3D point cloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_world)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        return pcd

    def _get_ee_camera_params(self):
        """
        Get end effector camera position and rotation matrix
        
        Returns:
        camera_pos: Camera position in world coordinates
        camera_R: Camera rotation matrix (from camera to world coordinates)
        """
        # End effector pose
        ee_pos, ee_orn = self.sim.robot.get_ee_pose()
        
        # End effector rotation matrix
        ee_R = np.array(p.getMatrixFromQuaternion(ee_orn)).reshape(3, 3)
        print("End effector orientation matrix:")
        print(ee_R)
        # Camera parameters
        cam_cfg = self.config["world_settings"]["camera"]
        ee_offset = np.array(cam_cfg["ee_cam_offset"])
        ee_cam_orn = cam_cfg["ee_cam_orientation"]
        ee_cam_R = np.array(p.getMatrixFromQuaternion(ee_cam_orn)).reshape(3, 3)
        # Calculate camera position
        camera_pos = ee_pos # Why does ee_pos + ee_R @ ee_offset cause an error?
        # Calculate camera rotation matrix
        camera_R = ee_R @ ee_cam_R
        
        return camera_pos, camera_R

    def visualize_point_clouds(self, collected_data, show_frames=True, show_merged=True):
        """
        Visualize collected point clouds using Open3D
        
        Parameters:
        collected_data: List of dictionaries containing point cloud data
        show_frames: Whether to show coordinate frames
        show_merged: Whether to show merged point cloud
        """
        if not collected_data:
            print("No point cloud data to visualize")
            return
            
        geometries = []
        
        # Add world coordinate frame
        if show_frames:
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
            geometries.append(coord_frame)
        
        if show_merged:
            # Merge point clouds using ICP
            print("Merging point clouds using ICP...")
            merged_pcd = self.merge_point_clouds(collected_data)
            if merged_pcd is not None:
                # Maintain original colors of the point cloud
                geometries.append(merged_pcd)
                print(f"Added merged point cloud with {len(merged_pcd.points)} points")
        else:
            # Add each point cloud and its camera coordinate frame
            for i, data in enumerate(collected_data):
                if 'point_cloud' in data and data['point_cloud'] is not None:
                    # Add point cloud
                    geometries.append(data['point_cloud'])
                    
                    # Add camera coordinate frame
                    if show_frames:
                        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                        camera_frame.translate(data['camera_position'])
                        camera_frame.rotate(data['camera_rotation'])
                        geometries.append(camera_frame)
                        
                    print(f"Added point cloud {i+1} with {len(data['point_cloud'].points)} points")
        
        print("Starting Open3D visualization...")
        o3d.visualization.draw_geometries(geometries)

    def merge_point_clouds(self, collected_data):
        """
        Merge multiple point clouds using ICP registration
        
        Parameters:
        collected_data: List of dictionaries containing point cloud data
        
        Returns:
        merged_pcd: Merged point cloud
        """
        if not collected_data:
            return None
            
        # Use first point cloud as reference
        merged_pcd = collected_data[0]['point_cloud']
        
        # ICP parameters
        threshold = 0.005  # Distance threshold
        trans_init = np.eye(4)  # Initial transformation
        
        # Merge remaining point clouds
        for i in range(1, len(collected_data)):
            current_pcd = collected_data[i]['point_cloud']
            
            # Execute ICP
            reg_p2p = o3d.pipelines.registration.registration_icp(
                current_pcd, merged_pcd, threshold, trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
            )
            
            # Transform current point cloud
            current_pcd.transform(reg_p2p.transformation)
            
            # Merge point clouds
            merged_pcd += current_pcd
            
            # Optional: Use voxel downsampling to remove duplicate points
            merged_pcd = merged_pcd.voxel_down_sample(voxel_size=0.005)
            
            print(f"Merged point cloud {i+1}, fitness: {reg_p2p.fitness}")
        
        return merged_pcd

    def collect_point_clouds(self, target_obj_name=None):
        """
        Main function to collect point clouds from multiple viewpoints
        
        Parameters:
        target_obj_name: Target object name, randomly selected if None
        
        Returns:
        collected_data: List of dictionaries containing point cloud data
        """
        print("Starting point cloud collection...")
        
        # If no target object name specified, randomly select one
        if target_obj_name is None:
            # Randomly select an object from YCB dataset
            object_root_path = self.sim.object_root_path
            files = glob.glob(os.path.join(object_root_path, "Ycb*"))
            obj_names = [os.path.basename(file) for file in files]
            target_obj_name = random.choice(obj_names)
            print(f"Resetting simulation with random object: {target_obj_name}")
        else:
            print(f"Resetting simulation with specified object: {target_obj_name}")
        
        # Reset simulation with target object
        self.sim.reset(target_obj_name)
        
        # Initialize point cloud collection list
        collected_data = []
        
        # Get and save initial position at simulation start
        initial_joints = self.sim.robot.get_joint_positions()
        print("Saving initial joint positions of simulation environment")
        
        # Initialize object height variable, default value
        object_height_with_offset = 1.6
        # Initialize object centroid coordinates, default values
        object_centroid_x = -0.02
        object_centroid_y = -0.45

        pause_time = 2.0  # Pause for 2 seconds
        print(f"\nPausing for {pause_time} seconds...")
        for _ in range(int(pause_time * 240)):  # Assuming simulation frequency of 240Hz
            self.sim.step()
            time.sleep(1/240.)
            
        # ===== Move to specified position and collect point cloud =====
        print("\nMoving to high observation point...")
        # Define high observation point position and orientation
        z_observe_pos = np.array([-0.02, -0.45, 1.9])
        z_observe_orn = p.getQuaternionFromEuler([0, np.radians(-180), 0])  # Looking down
        
        # Solve IK
        from src.ik_solver import DifferentialIKSolver
        ik_solver = DifferentialIKSolver(self.sim.robot.id, self.sim.robot.ee_idx, damping=0.05)
        high_point_target_joints = ik_solver.solve(z_observe_pos, z_observe_orn, initial_joints, max_iters=50, tolerance=0.001)
        
        # Generate trajectory
        print("Generating trajectory for high observation point...")
        high_point_trajectory = SimpleTrajectoryPlanner.generate_joint_trajectory(initial_joints, high_point_target_joints, steps=100)
        if not high_point_trajectory:
            print("Unable to generate trajectory to high observation point, skipping high point cloud collection")
        else:
            print(f"Generated trajectory with {len(high_point_trajectory)} points")
            
            # Reset to initial position
            for i, joint_idx in enumerate(self.sim.robot.arm_idx):
                p.resetJointState(self.sim.robot.id, joint_idx, initial_joints[i])
            
            # Move robot along trajectory to high point
            for joint_target in high_point_trajectory:
                self.sim.robot.position_control(joint_target)
                for _ in range(1):
                    self.sim.step()
                    time.sleep(1/240.)
            
            # Collect point cloud at high observation point
            rgb_ee, depth_ee, seg_ee = self.sim.get_ee_renders()
            camera_pos, camera_R = self._get_ee_camera_params()
            print(f"High observation point camera position:", camera_pos)
            print(f"High observation point end effector position:", self.sim.robot.get_ee_pose()[0])
            
            # Build point cloud
            target_mask_id = self.sim.object.id
            print(f"Target object ID: {target_mask_id}")
            
            try:
                if target_mask_id not in np.unique(seg_ee):
                    print("Warning: Target object ID not found in segmentation mask")
                    print("Available IDs in segmentation mask:", np.unique(seg_ee))
                    
                    non_zero_ids = np.unique(seg_ee)[1:] if len(np.unique(seg_ee)) > 1 else []
                    if len(non_zero_ids) > 0:
                        target_mask_id = non_zero_ids[0]
                        print(f"Using first non-zero ID instead: {target_mask_id}")
                    else:
                        raise ValueError("No valid objects found in segmentation mask")
                
                high_point_pcd = self._build_object_point_cloud_ee(rgb_ee, depth_ee, seg_ee, target_mask_id, camera_pos, camera_R)
                
                # Process point cloud
                high_point_pcd = high_point_pcd.voxel_down_sample(voxel_size=0.005)
                high_point_pcd, _ = high_point_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
                
                # Store point cloud data
                high_point_cloud_data = {
                    'point_cloud': high_point_pcd,
                    'camera_position': camera_pos,
                    'camera_rotation': camera_R,
                    'ee_position': self.sim.robot.get_ee_pose()[0],
                    'timestamp': time.time(),
                    'target_object': target_obj_name,
                    'viewpoint_idx': 'high_point'
                }
                
                # Get coordinates of all points in the point cloud
                points_array = np.asarray(high_point_pcd.points)
                if len(points_array) > 0:
                    # Find maximum z-axis point
                    max_z_idx = np.argmax(points_array[:, 2])
                    max_z_point = points_array[max_z_idx]
                    print(f"Maximum z-axis point in high point cloud: {max_z_point}")
                    high_point_cloud_data['max_z_point'] = max_z_point
                    
                    # Extract maximum z-axis value, add offset
                    object_max_z = max_z_point[2]
                    object_height_with_offset = max(object_max_z + 0.2, 1.65)
                    print(f"Object height with offset: {object_height_with_offset}")
                    
                    # Calculate centroid of x and y coordinates of all points in the cloud
                    object_centroid_x = np.mean(points_array[:, 0])
                    object_centroid_y = np.mean(points_array[:, 1])
                    print(f"Object point cloud centroid coordinates (x, y): ({object_centroid_x:.4f}, {object_centroid_y:.4f})")
                    high_point_cloud_data['centroid'] = np.array([object_centroid_x, object_centroid_y, 0])
                else:
                    print("No points in high point cloud")
                
                # Add high point cloud to collected data
                collected_data.append(high_point_cloud_data)
                print(f"Collected point cloud from high observation point with {len(high_point_pcd.points)} points")
                
            except ValueError as e:
                print(f"Error building point cloud for high observation point:", e)

        # Dynamically generate target positions and orientations based on object centroid coordinates
        # Determine if object is far from the robot arm (x<-0.2 and y<-0.5 considered far)
        is_object_far = object_centroid_x < -0.2 and object_centroid_y < -0.5
        
        # Basic sampling directions
        target_positions = []
        target_orientations = []
        
        # East direction
        target_positions.append(np.array([object_centroid_x + 0.15, object_centroid_y, object_height_with_offset]))
        target_orientations.append(p.getQuaternionFromEuler([0, np.radians(-150), 0]))
        
        # North direction
        target_positions.append(np.array([object_centroid_x, object_centroid_y + 0.15, object_height_with_offset]))
        target_orientations.append(p.getQuaternionFromEuler([np.radians(150), 0, 0]))
        
        # West direction
        target_positions.append(np.array([object_centroid_x - 0.15, object_centroid_y, object_height_with_offset]))
        target_orientations.append(p.getQuaternionFromEuler([0, np.radians(150), 0]))
        
        # South direction (add only if object is not far away)
        if not is_object_far:
            target_positions.append(np.array([object_centroid_x, object_centroid_y - 0.15, object_height_with_offset]))
            target_orientations.append(p.getQuaternionFromEuler([np.radians(-150), 0, 0]))
        else:
            print("Object position is far (x<-0.2 and y<-0.5), skipping south direction sampling point to avoid singular points")
        
        # Top view
        target_positions.append(np.array([-0.02, -0.45, 1.8]))
        target_orientations.append(p.getQuaternionFromEuler([np.radians(180), 0, np.radians(-90)]))
        
        print(f"\nUsing collection positions based on object centroid:")
        print(f"Object centroid coordinates (x, y): ({object_centroid_x:.4f}, {object_centroid_y:.4f})")
        print(f"Object height with offset: {object_height_with_offset:.4f}")
        for i, pos in enumerate(target_positions):
            print(f"Collection point {i+1}: ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
        
        # Collect from each viewpoint
        for viewpoint_idx, (target_pos, target_orn) in enumerate(zip(target_positions, target_orientations)):
            print(f"\nMoving to viewpoint {viewpoint_idx + 1}")
            self.sim.get_ee_renders()
            
            # Get current joint positions
            current_joints = self.sim.robot.get_joint_positions()
            # Save current joint positions
            saved_joints = current_joints.copy()
            
            # Solve IK for target end effector pose
            target_joints = ik_solver.solve(target_pos, target_orn, current_joints, max_iters=50, tolerance=0.001)
            
            # Reset to saved starting position
            for i, joint_idx in enumerate(self.sim.robot.arm_idx):
                p.resetJointState(self.sim.robot.id, joint_idx, saved_joints[i])
            
            # Choose trajectory generation method
            choice = 2  # Change this value to test different methods
            
            trajectory = []
            if choice == 1:
                print("Generating linear Cartesian trajectory...")
                trajectory = SimpleTrajectoryPlanner.generate_cartesian_trajectory(
                    self.sim.robot.id, 
                    self.sim.robot.arm_idx, 
                    self.sim.robot.ee_idx, 
                    saved_joints, 
                    target_pos, 
                    target_orn, 
                    steps=100
                )
            elif choice == 2:
                print("Generating linear joint space trajectory...")
                trajectory = SimpleTrajectoryPlanner.generate_joint_trajectory(saved_joints, target_joints, steps=100)
            
            if not trajectory:
                print(f"Unable to generate trajectory for viewpoint {viewpoint_idx + 1}. Skipping...")
                continue
            
            print(f"Generated trajectory with {len(trajectory)} points")
            
            # Reset to saved starting position again before executing trajectory
            for i, joint_idx in enumerate(self.sim.robot.arm_idx):
                p.resetJointState(self.sim.robot.id, joint_idx, saved_joints[i])
            
            # Move robot along trajectory to target position
            for joint_target in trajectory:
                # Move robot
                self.sim.robot.position_control(joint_target)
                for _ in range(1):
                    self.sim.step()
                    time.sleep(1/240.)
            
            # Capture point cloud at this viewpoint
            rgb_ee, depth_ee, seg_ee = self.sim.get_ee_renders()
            camera_pos, camera_R = self._get_ee_camera_params()
            print(f"Viewpoint {viewpoint_idx + 1} camera position:", camera_pos)
            print(f"Viewpoint {viewpoint_idx + 1} end effector position:", self.sim.robot.get_ee_pose()[0])
            
            # Build point cloud
            target_mask_id = self.sim.object.id
            print(f"Target object ID: {target_mask_id}")
            
            try:
                if target_mask_id not in np.unique(seg_ee):
                    print("Warning: Target object ID not found in segmentation mask")
                    print("Available IDs in segmentation mask:", np.unique(seg_ee))
                    
                    non_zero_ids = np.unique(seg_ee)[1:] if len(np.unique(seg_ee)) > 1 else []
                    if len(non_zero_ids) > 0:
                        target_mask_id = non_zero_ids[0]
                        print(f"Using first non-zero ID instead: {target_mask_id}")
                    else:
                        raise ValueError("No valid objects found in segmentation mask")
                
                pcd_ee = self._build_object_point_cloud_ee(rgb_ee, depth_ee, seg_ee, target_mask_id, camera_pos, camera_R)
                
                # Process point cloud
                pcd_ee = pcd_ee.voxel_down_sample(voxel_size=0.005)
                pcd_ee, _ = pcd_ee.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
                
                # Store point cloud data
                point_cloud_data = {
                    'point_cloud': pcd_ee,
                    'camera_position': camera_pos,
                    'camera_rotation': camera_R,
                    'ee_position': self.sim.robot.get_ee_pose()[0],
                    'timestamp': time.time(),
                    'target_object': target_obj_name,
                    'viewpoint_idx': viewpoint_idx
                }
                collected_data.append(point_cloud_data)
                print(f"Collected point cloud from viewpoint {viewpoint_idx + 1} with {len(pcd_ee.points)} points.")
                
            except ValueError as e:
                print(f"Error building point cloud for viewpoint {viewpoint_idx + 1}:", e)

        return collected_data
