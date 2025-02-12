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
from src.point_cloud import build_object_point_cloud
from src.ik_solver import DifferentialIKSolver

def print_segmentation_info(seg):
    """
    打印 segmentation mask 中包含的所有唯一 id 信息，并尝试通过 p.getBodyInfo() 查询对应物体名称。
    
    注意：一般 0 表示背景，其余 id 经过编码后低 24 位为 body id。
    """
    unique_ids = np.unique(seg)
    print("Segmentation mask 中的唯一 id:")
    for seg_id in unique_ids:
        if seg_id == 0:
            print(f"  ID {seg_id}: 背景")
        else:
            body_id = int(seg_id) & ((1 << 24) - 1)
            try:
                body_info = p.getBodyInfo(body_id)
                body_name = body_info[0].decode("utf-8") if body_info[0] is not None else "未知"
                print(f"  Segmentation ID {seg_id} -> body id {body_id}: 名称 = {body_name}")
            except Exception as e:
                print(f"  Segmentation ID {seg_id} -> body id {body_id}: 无法获取信息, 错误: {e}")

def run_point_cloud_visualization(config):
    print("Starting interactive point cloud visualization ...")
    
    # 初始化仿真（带 GUI）
    sim = Simulation(config)
    
    # 从 YCB 数据集中随机选一个物体
    object_root_path = ycb_objects.getDataPath()
    files = glob.glob(os.path.join(object_root_path, "Ycb*"))
    obj_names = [os.path.basename(file) for file in files]
    target_obj_name = random.choice(obj_names)
    print("Resetting simulation with random object:", target_obj_name)
    
    # 重置仿真并加载物体
    sim.reset(target_obj_name)
    time.sleep(1)  # 等待仿真稳定

    # === 1. 采集高空静态摄像头图像，生成点云 ===
    print("Capturing images from static (overhead) camera ...")
    rgb_static, depth_static, seg_static = sim.get_static_renders()
    print("Static camera segmentation info:")
    print_segmentation_info(seg_static)
    
    # 静态摄像头中目标物体的 mask id 为 5
    target_mask_static = 5
    try:
        pcd_static = build_object_point_cloud(rgb_static, depth_static, seg_static, target_mask_static, config)
    except ValueError as e:
        print("Error building point cloud from static camera:", e)
        pcd_static = None

    # === 2. 移动机械臂，使得 end-effector 摄像头对准物体 ===
    print("Moving robot for end-effector view ...")
    # 目标末端执行器位姿（数值可根据实际情况调整）
    target_pos = np.array([-0.2, -0.45, 1.5])
    target_orn = [0, 1, 0, -0.02937438]
    
    # 获取当前机械臂关节角
    current_joints = sim.robot.get_joint_positions()
    
    # 利用 DifferentialIKSolver 求解目标末端执行器位姿对应的关节角
    ik_solver = DifferentialIKSolver(sim.robot.id, sim.robot.ee_idx, damping=0.05)
    print("Solving IK for target end-effector pose ...")
    new_joints = ik_solver.solve(target_pos, target_orn, current_joints, max_iters=50, tolerance=0.01)
    
    # 下发关节角控制命令
    print("Moving robot to new joint configuration ...")
    sim.robot.position_control(new_joints)
    for _ in range(500):  # 运行一段时间让机械臂运动到位
        p.stepSimulation()
        time.sleep(1/240.)
    
    # === 3. 采集机械臂同轴摄像头图像，生成点云 ===
    print("Capturing images from end-effector camera ...")
    rgb_ee, depth_ee, seg_ee = sim.get_ee_renders()
    print("End-effector camera segmentation info:")
    print_segmentation_info(seg_ee)
    
    # 计算机械臂摄像头的外参：
    # 1. 获取末端执行器的位姿
    ee_pos, ee_ori = sim.robot.get_ee_pose()  # ee_pos: [x,y,z], ee_ori: 四元数
    # 2. 从配置中获取 ee_cam_offset，并计算旋转矩阵
    ee_cam_offset = np.array(config["world_settings"]["camera"]["ee_cam_offset"])
    R_ee = np.array(p.getMatrixFromQuaternion(ee_ori)).reshape(3, 3)
    # 3. 计算摄像头在世界坐标系下的位置
    ee_cam_pos = np.array(ee_pos) + R_ee.dot(ee_cam_offset)
    # 4. 定义摄像头目标点：假设摄像头局部坐标系下正 z 轴为前向
    forward_vector = R_ee.dot(np.array([0, 0, 1]))
    ee_target_pos = ee_cam_pos + 0.1 * forward_vector  # 例如前进 0.1 米
    
    # 机械臂摄像头中目标物体的 mask id 同样为 5
    target_mask_ee = 5
    try:
        pcd_ee = build_object_point_cloud(
            rgb_ee, depth_ee, seg_ee, target_mask_ee, config,
            cam_pos=ee_cam_pos, target_pos=ee_target_pos
        )
    except ValueError as e:
        print("Error building point cloud from end-effector camera:", e)
        pcd_ee = None
    
    # === 4. 合并两路点云并在同一交互窗口中显示 ===
    # 创建世界坐标系坐标轴
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    
    geometries = [coord_frame]
    if pcd_static is not None:
        geometries.append(pcd_static)
    if pcd_ee is not None:
        geometries.append(pcd_ee)
    
    print("Launching interactive Open3D visualization with merged point clouds ...")
    o3d.visualization.draw_geometries(geometries)
    
    sim.close()

if __name__ == "__main__":
    with open("configs/test_config.yaml", "r") as stream:
        config = yaml.safe_load(stream)
    run_point_cloud_visualization(config)
