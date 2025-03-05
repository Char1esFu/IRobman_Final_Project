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

def get_intrinsic_matrix(width, height, fov):
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
    world_points_homogeneous = np.dot(points_homogeneous, camera_extrinsic.T) # points in rows
    
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
    intrinsic_matrix = get_intrinsic_matrix(width, height, fov)
    
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
    print("End effector camera orientation matrix:")
    print(ee_cam_R)
    # calculate camera position
    camera_pos = ee_pos + ee_R @ ee_offset
    
    # calculate camera rotation matrix
    camera_R = ee_R @ ee_cam_R
    
    return camera_pos, camera_R

def generate_trajectory(start_joints, end_joints, steps=100):
    """
    generate smooth trajectory from start to end joint positions
    
    Parameters:
    start_joints: start joint positions
    end_joints: end joint positions
    steps: number of steps for interpolation
    
    Returns:
    trajectory: list of joint positions
    """
    trajectory = []
    for step in range(steps + 1):
        t = step / steps  # normalize step
        # linear interpolation
        point = [start + t * (end - start) for start, end in zip(start_joints, end_joints)]
        trajectory.append(point)
    return trajectory

def run_point_cloud_visualization(config):
    """
    main function to run interactive point cloud visualization
    """
    print("Starting interactive point cloud visualization ...")
    
    # initialize PyBullet simulation
    sim = Simulation(config)
    
    # randomly select an object from YCB dataset
    object_root_path = ycb_objects.getDataPath()
    files = glob.glob(os.path.join(object_root_path, "Ycb*"))
    obj_names = [os.path.basename(file) for file in files]
    target_obj_name = random.choice(obj_names)
    print(f"Resetting simulation with random object: {target_obj_name}")
    
    # reset simulation with target object
    sim.reset(target_obj_name)
    time.sleep(1)  # wait for objects to settle
    
    # 1. move robot to target position
    target_pos = np.array([-0.2, -0.45, 1.7])
    target_orn = p.getQuaternionFromEuler([0, np.radians(135), 0])
    
    # get current joint positions
    current_joints = sim.robot.get_joint_positions()
    # save current joint positions
    saved_joints = current_joints.copy()
    
    # solve IK for target end-effector pose
    ik_solver = DifferentialIKSolver(sim.robot.id, sim.robot.ee_idx, damping=0.05)
    new_joints = ik_solver.solve(target_pos, target_orn, current_joints, max_iters=50, tolerance=0.01)
    
    # reset to saved start position
    for i, joint_idx in enumerate(ik_solver.joint_indices):
        p.resetJointState(sim.robot.id, joint_idx, saved_joints[i])
    
    # linear interpolation between start and end joint positions, rrt or other path planning algorithms can be used    
    trajectory = generate_trajectory(current_joints, new_joints, steps=100)
    # move robot along trajectory to target position
    for joint_target in trajectory:
        sim.robot.position_control(joint_target)
        for _ in range(1):  
            sim.step()
            time.sleep(1/240.)  # 240 Hz
            
    # 2. capture images from end-effector camera
    print("Capturing images from end-effector camera ...")
    rgb_ee, depth_ee, seg_ee = sim.get_ee_renders()
    
    # get camera parameters
    camera_pos, camera_R = get_ee_camera_params(sim.robot, config)
    print("Camera position in world frame:", camera_pos)
    print("End effector position in world frame:", sim.robot.get_ee_pose()[0])
    
    # 3. build point cloud from end-effector camera data
    target_mask_id = sim.object.id  # target object ID
    print(f"Target object ID: {target_mask_id}")
    
    try:
        # search for target object ID in segmentation mask
        if target_mask_id not in np.unique(seg_ee):
            print("Warning: Target object ID not found in segmentation mask.")
            print("Available IDs in segmentation mask:", np.unique(seg_ee))
            
            # use first non-zero ID as target mask ID
            non_zero_ids = np.unique(seg_ee)[1:] if len(np.unique(seg_ee)) > 1 else []
            if len(non_zero_ids) > 0:
                target_mask_id = non_zero_ids[0]
                print(f"Using first non-zero ID instead: {target_mask_id}")
            else:
                raise ValueError("No valid objects found in segmentation mask")
        
        pcd_ee = build_object_point_cloud_ee(rgb_ee, depth_ee, seg_ee, target_mask_id, config, camera_pos, camera_R)
        
        # optional: downsample point cloud
        pcd_ee = pcd_ee.voxel_down_sample(voxel_size=0.005)
        
        # optional: remove statistical outliers
        pcd_ee, _ = pcd_ee.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
    except ValueError as e:
        print("Error building point cloud:", e)
        pcd_ee = None
    
    # 4. visualize point cloud and camera position
    # create world coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    
    # create camera coordinate frame
    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    camera_frame.translate(camera_pos)
    camera_frame.rotate(camera_R)
    
    # show point cloud and camera frame
    geometries = [coord_frame, camera_frame]
    if pcd_ee is not None:
        geometries.append(pcd_ee)
        print(f"Point cloud created with {len(pcd_ee.points)} points.")
    
    print("Launching interactive Open3D visualization ...")
    o3d.visualization.draw_geometries(geometries)
    
    sim.close()


if __name__ == "__main__":
    with open("configs/test_config.yaml", "r") as stream:
        config = yaml.safe_load(stream)
    run_point_cloud_visualization(config)