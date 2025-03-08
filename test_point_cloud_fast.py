import os
import glob
import yaml
import time
import random
import numpy as np
import pybullet as p
import open3d as o3d
from pybullet_object_models import ycb_objects  # type:ignore
from src.simulation import Simulation
from src.ik_solver import DifferentialIKSolver
from src.obstacle_tracker import ObstacleTracker
from src.rrt_star import RRTStarPlanner

# Check if PyBullet has NumPy support enabled
numpy_support = p.isNumpyEnabled()
print(f"PyBullet NumPy support enabled: {numpy_support}")

def convert_depth_to_meters(depth_buffer, near, far):
    """
    convert depth buffer values to actual distance (meters)
    
    Parameters:
    depth_buffer: depth buffer values obtained from PyBullet
    near, far: near/far plane distances
    
    Returns:
    actual depth values in meters
    """
    return far * near / (far - (far - near) * depth_buffer)

def get_camera_intrinsic(width, height, fov):
    """
    calculate intrinsic matrix from camera parameters
    
    Parameters:
    width: image width (pixels)
    height: image height (pixels)
    fov: vertical field of view (degrees)

    Returns:
    camera intrinsic matrix
    """    
    # calculate focal length
    f = height / (2 * np.tan(np.radians(fov / 2)))
    
    # calculate principal point
    cx = width / 2
    cy = height / 2
    
    intrinsic_matrix = np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0, 1]
    ])
    
    return intrinsic_matrix

def depth_image_to_point_cloud(depth_image, mask, rgb_image, intrinsic_matrix):
    """
    depth image to camera coordinate point cloud
    
    Parameters:
    depth_image: depth image (meters)
    mask: target object mask (boolean array)
    rgb_image: RGB image
    intrinsic_matrix: camera intrinsic matrix
    
    Returns:
    camera coordinate point cloud (N,3) and corresponding colors (N,3)
    """
    # extract pixel coordinates of target mask
    rows, cols = np.where(mask)
    
    if len(rows) == 0:
        raise ValueError("No valid pixels found in target mask")
    
    # extract depth values of these pixels
    depths = depth_image[rows, cols]
    
    # image coordinates to camera coordinates
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    
    # calculate camera coordinates
    x = -(cols - cx) * depths / fx # negative sign due to PyBullet camera orientation???
    y = -(rows - cy) * depths / fy
    z = depths
    
    # stack points
    points = np.vstack((x, y, z)).T
    
    # extract RGB colors
    colors = rgb_image[rows, cols, :3].astype(np.float64) / 255.0
    
    return points, colors

def transform_points_to_world(points, camera_extrinsic):
    """
    transform points from camera coordinates to world coordinates
    
    Parameters:
    points: point cloud in camera coordinates (N,3)
    camera_extrinsic: camera extrinsic matrix (4x4)
    
    Returns:
    point cloud in world coordinates (N,3)
    """
    # convert point cloud to homogeneous coordinates
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    
    # transform point cloud using extrinsic matrix
    world_points_homogeneous = (camera_extrinsic @ points_homogeneous.T).T # points in rows
    
    # convert back to non-homogeneous coordinates
    world_points = world_points_homogeneous[:, :3]
    
    return world_points

def get_camera_extrinsic(camera_pos, camera_R):
    """
    build camera extrinsic matrix (transform from camera to world coordinates)
    
    Parameters:
    camera_pos: camera position in world coordinates
    camera_R: camera rotation matrix (3x3)
    
    Returns:
    camera extrinsic matrix (4x4)
    """
    # build 4x4 extrinsic matrix
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = camera_R
    extrinsic[:3, 3] = camera_pos
    
    return extrinsic

def build_object_point_cloud_ee(rgb, depth, seg, target_mask_id, config, camera_pos, camera_R):
    """
    build object point cloud using end-effector camera RGB, depth, segmentation data
    
    Parameters:
    rgb: RGB image
    depth: depth buffer values
    seg: segmentation mask
    target_mask_id: target object ID
    config: configuration dictionary
    camera_pos: camera position in world coordinates
    camera_R: camera rotation matrix (from camera to world coordinates)
    
    Returns:
    Open3D point cloud object
    """
    # read camera parameters
    cam_cfg = config["world_settings"]["camera"]
    width = cam_cfg["width"]
    height = cam_cfg["height"]
    fov = cam_cfg["fov"]  # vertical FOV
    near = cam_cfg["near"]
    far = cam_cfg["far"]
    
    # create target object mask
    object_mask = (seg == target_mask_id)
    if np.count_nonzero(object_mask) == 0:
        raise ValueError(f"Target mask ID {target_mask_id} not found in segmentation.")
    
    # extract depth buffer values for target object
    metric_depth = convert_depth_to_meters(depth, near, far)
    
    # get intrinsic matrix
    intrinsic_matrix = get_camera_intrinsic(width, height, fov)
    
    # convert depth image to point cloud
    points_cam, colors = depth_image_to_point_cloud(metric_depth, object_mask, rgb, intrinsic_matrix)
    
    # build camera extrinsic matrix
    camera_extrinsic = get_camera_extrinsic(camera_pos, camera_R)
    
    # transform points to world coordinates
    points_world = transform_points_to_world(points_cam, camera_extrinsic)
    
    # create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_world)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd

def get_ee_camera_params(robot, config):
    """
    get end-effector camera position and rotation matrix
    
    Parameters:
    robot: robot object
    config: configuration dictionary
    
    Returns:
    camera_pos: camera position in world coordinates
    camera_R: camera rotation matrix (from camera to world coordinates)
    """
    # end-effector pose
    ee_pos, ee_orn = robot.get_ee_pose()
    
    # end-effector rotation matrix
    ee_R = np.array(p.getMatrixFromQuaternion(ee_orn)).reshape(3, 3)
    print("End effector orientation matrix:")
    print(ee_R)
    # camera parameters
    cam_cfg = config["world_settings"]["camera"]
    ee_offset = np.array(cam_cfg["ee_cam_offset"])
    ee_cam_orn = cam_cfg["ee_cam_orientation"]
    ee_cam_R = np.array(p.getMatrixFromQuaternion(ee_cam_orn)).reshape(3, 3)
    # calculate camera position
    camera_pos = ee_pos # why ee_pos + ee_R @ ee_offset will be wrong?
    # calculate camera rotation matrix
    camera_R = ee_R @ ee_cam_R
    
    return camera_pos, camera_R

def iterative_closest_point(collected_data):
    """
    Merge multiple point clouds using ICP registration
    
    Parameters:
    collected_data: list of dictionaries containing point cloud data
    
    Returns:
    merged_pcd: merged point cloud
    """
    if not collected_data:
        return None
        
    # Use the first point cloud as reference
    merged_pcd = collected_data[0]['point_cloud']
    
    # ICP parameters
    threshold = 0.005  # distance threshold
    trans_init = np.eye(4)  # initial transformation
    
    # Merge remaining point clouds
    for i in range(1, len(collected_data)):
        current_pcd = collected_data[i]['point_cloud']
        
        # Perform ICP
        reg_p2p = o3d.pipelines.registration.registration_icp(
            current_pcd, merged_pcd, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
        )
        
        # Transform current point cloud
        current_pcd.transform(reg_p2p.transformation)
        
        # Merge point clouds
        merged_pcd += current_pcd
        
        # Optional: Remove duplicates using voxel downsampling
        merged_pcd = merged_pcd.voxel_down_sample(voxel_size=0.005)
        
        print(f"Merged point cloud {i+1}, fitness: {reg_p2p.fitness}")
    
    return merged_pcd

def visualize_point_clouds(collected_data, show_frames=True, show_merged=True):
    """
    Visualize collected point clouds using Open3D
    
    Parameters:
    collected_data: list of dictionaries containing point cloud data
    show_frames: whether to show coordinate frames
    show_merged: whether to show merged point cloud
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
        merged_pcd = iterative_closest_point(collected_data)
        if merged_pcd is not None:
            # Keep original colors from point clouds
            geometries.append(merged_pcd)
            print(f"Added merged point cloud with {len(merged_pcd.points)} points")
    else:
        # Add each point cloud and its camera frame
        for i, data in enumerate(collected_data):
            if 'point_cloud' in data and data['point_cloud'] is not None:
                # Add point cloud
                geometries.append(data['point_cloud'])
                
                # Add camera frame
                if show_frames:
                    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                    camera_frame.translate(data['camera_position'])
                    camera_frame.rotate(data['camera_rotation'])
                    geometries.append(camera_frame)
                    
                print(f"Added point cloud {i+1} with {len(data['point_cloud'].points)} points")
    
    print("Launching Open3D visualization...")
    o3d.visualization.draw_geometries(geometries)

def run(config):
    """
    main function to run point cloud collection from multiple viewpoints
    """
    print("Starting point cloud collection ...")
    
    # initialize PyBullet simulation
    sim = Simulation(config)
    
    # randomly select an object from YCB dataset
    object_root_path = ycb_objects.getDataPath()
    files = glob.glob(os.path.join(object_root_path, "Ycb*"))
    obj_names = [os.path.basename(file) for file in files]
    # target_obj_name = random.choice(obj_names)
    # print(f"Resetting simulation with random object: {target_obj_name}")
    # All objects: 
    # Low objects: YcbBanana, YcbFoamBrick, YcbHammer, YcbMediumClamp, YcbPear, YcbScissors, YcbStrawberry, YcbTennisBall, 
    # Medium objects: YcbGelatinBox, YcbMasterChefCan, YcbPottedMeatCan, YcbTomatoSoupCan
    # High objects: YcbCrackerBox, YcbMustardBottle, 
    # Unstable objects: YcbChipsCan， YcbPowerDrill
    target_obj_name = "YcbTennisBall" 
    
    # reset simulation with target object
    sim.reset(target_obj_name)
    
    # Wait for objects to settle before collecting point clouds
    print("Waiting for objects to settle...")
    for _ in range(100):
        sim.step()
        time.sleep(1/240.)
    print("Objects settled, starting point cloud collection")
    
    # Initialize obstacle tracker
    obstacle_tracker = ObstacleTracker(n_obstacles=2, exp_settings=config)
    
    # Initialize point cloud collection list
    collected_data = []
    
    # Define target positions and orientations
    target_positions = [
        # 3 viewpoints
        np.array([-0.015, -0.3, 1.6]),
        np.array([-0.215, -0.6, 1.6]),
        np.array([0.185, -0.6, 1.6]),
        # 4 viewpoints
        # np.array([-0.02, -0.3, 1.6]),
        # np.array([-0.02, -0.6, 1.6]),
        # np.array([-0.22, -0.45, 1.6]),
        # np.array([0.18, -0.45, 1.6]),

    ]
    target_orientations = [
        # 3 viewpoints
        p.getQuaternionFromEuler([np.radians(140), 0, 0]),
        p.getQuaternionFromEuler([np.radians(-150), np.radians(-30), 0]), 
        p.getQuaternionFromEuler([np.radians(-150), np.radians(30), 0]),
        # 4 viewpoints
        # p.getQuaternionFromEuler([np.radians(135), 0, 0]),
        # p.getQuaternionFromEuler([np.radians(-135), 0, 0]),
        # p.getQuaternionFromEuler([0, np.radians(135), 0]), 
        # p.getQuaternionFromEuler([0, np.radians(-135), 0])
    ]
    
    # For each viewpoint
    for viewpoint_idx, (target_pos, target_orn) in enumerate(zip(target_positions, target_orientations)):
        print(f"\nMoving to viewpoint {viewpoint_idx + 1}")
        
        # Get initial static camera view to setup obstacle tracking
        rgb_static, depth_static, seg_static = sim.get_static_renders()
        detections = obstacle_tracker.detect_obstacles(rgb_static, depth_static, seg_static)
        tracked_positions = obstacle_tracker.update(detections)
        
        # Get current joint positions
        current_joints = sim.robot.get_joint_positions()
        
        # Solve IK for target end-effector pose
        ik_solver = DifferentialIKSolver(sim.robot.id, sim.robot.ee_idx, damping=0.05)
        target_joints = ik_solver.solve(target_pos, target_orn, current_joints, max_iters=50, tolerance=0.01)
        
        # Directly set robot to target joints
        sim.robot.position_control(target_joints)
        for _ in range(5):  # Give some time for physics to settle
            sim.step()
            time.sleep(1/240.)
        
        # Capture point cloud at this viewpoint
        rgb_ee, depth_ee, seg_ee = sim.get_ee_renders()
        camera_pos, camera_R = get_ee_camera_params(sim.robot, config)
        print(f"Viewpoint {viewpoint_idx + 1} camera position:", camera_pos)
        print(f"Viewpoint {viewpoint_idx + 1} end effector position:", sim.robot.get_ee_pose()[0])
        
        # Build point cloud
        target_mask_id = sim.object.id
        print(f"Target object ID: {target_mask_id}")
        
        try:
            if target_mask_id not in np.unique(seg_ee):
                print("Warning: Target object ID not found in segmentation mask.")
                print("Available IDs in segmentation mask:", np.unique(seg_ee))
                
                non_zero_ids = np.unique(seg_ee)[1:] if len(np.unique(seg_ee)) > 1 else []
                if len(non_zero_ids) > 0:
                    target_mask_id = non_zero_ids[0]
                    print(f"Using first non-zero ID instead: {target_mask_id}")
                else:
                    raise ValueError("No valid objects found in segmentation mask")
            
            pcd_ee = build_object_point_cloud_ee(rgb_ee, depth_ee, seg_ee, target_mask_id, config, camera_pos, camera_R)
            
            # Process point cloud
            pcd_ee = pcd_ee.voxel_down_sample(voxel_size=0.005)
            pcd_ee, _ = pcd_ee.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            
            # Store point cloud data
            point_cloud_data = {
                'point_cloud': pcd_ee,
                'camera_position': camera_pos,
                'camera_rotation': camera_R,
                'ee_position': sim.robot.get_ee_pose()[0],
                'timestamp': time.time(),
                'target_object': target_obj_name,
                'viewpoint_idx': viewpoint_idx
            }
            collected_data.append(point_cloud_data)
            print(f"Point cloud collected from viewpoint {viewpoint_idx + 1} with {len(pcd_ee.points)} points.")
            
        except ValueError as e:
            print(f"Error building point cloud for viewpoint {viewpoint_idx + 1}:", e)
    
    sim.close()
    return collected_data

if __name__ == "__main__":
    with open("configs/test_config.yaml", "r") as stream:
        config = yaml.safe_load(stream)
    # Run simulation and collect point clouds
    collected_point_clouds = run(config)
    print(f"Successfully collected {len(collected_point_clouds)} point clouds.")
    
    # Visualize the collected point clouds if any were collected
    if collected_point_clouds:
        # First show individual point clouds
        print("\nVisualizing individual point clouds...")
        visualize_point_clouds(collected_point_clouds, show_merged=False)
        
        # Then show merged point cloud
        print("\nVisualizing merged point cloud...")
        visualize_point_clouds(collected_point_clouds, show_merged=True)