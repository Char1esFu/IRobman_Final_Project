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
    打印 segmentation mask 中的所有唯一 id 信息，并尝试查询对应物体名称。
    注意：一般0为背景，其余 id 经过编码后低 24 位为 body id。
    """
    unique_ids = np.unique(seg)
    print("Segmentation mask 中的唯一 id:")
    for seg_id in unique_ids:
        if seg_id == 0:
            print(f"  ID {seg_id}: 背景")
        else:
            # 提取低 24 位作为 body id
            body_id = int(seg_id) & ((1 << 24) - 1)
            try:
                body_info = p.getBodyInfo(body_id)
                # p.getBodyInfo 返回 tuple，第一个元素一般为物体名称（bytes 类型）
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
    
    # === 1. 机械臂移动至预定区域 ===
    # 目标末端执行器位姿（数值不需要特别精确）
    target_pos = np.array([-0.2, -0.45, 1.5])
    target_orn = [1, 0, 0, 0]
    
    # 获取当前机械臂关节角
    current_joints = sim.robot.get_joint_positions()
    
    # 创建 IK 求解器
    ik_solver = DifferentialIKSolver(sim.robot.id, sim.robot.ee_idx, damping=0.05)
    print("Solving IK for target end-effector pose ...")
    new_joints = ik_solver.solve(target_pos, target_orn, current_joints, max_iters=50, tolerance=0.01)
    
    # 发送目标关节角，控制机械臂运动
    print("Moving robot to new joint configuration ...")
    sim.robot.position_control(new_joints)
    
    # 运行一段时间让机械臂运动到位（这里用 500 个仿真步作为示例）
    for _ in range(500):
        p.stepSimulation()
        time.sleep(1/240.)
    
    # === 2. 采集机械臂同轴摄像头图像 ===
    print("Capturing images from end-effector camera ...")
    rgb_ee, depth_ee, seg_ee = sim.get_ee_renders()

    # 调用打印函数，查看 segmentation mask 中的 id 信息
    print_segmentation_info(seg_ee)

    # === 计算机械臂同轴摄像头的外参 extrinsics ===
    # 假设 sim.robot.get_ee_pose() 返回 (position, orientation)
    # position 为 [x, y, z]，orientation 为四元数 [x, y, z, w]
    ee_pos, ee_orn = sim.robot.get_ee_pose()

    # 将末端执行器的四元数转换为旋转矩阵（3x3）
    R_ee = np.array(p.getMatrixFromQuaternion(ee_orn)).reshape(3, 3)
    print("End effector orientation matrix:", R_ee)
    # 从配置文件中读取摄像头相对于末端执行器的 offset 与旋转（四元数）
    cam_cfg = config["world_settings"]["camera"]
    ee_offset = np.array(cam_cfg["ee_cam_offset"])      # 例如 [0.0, 0.0, 0.1]
    ee_cam_orn = cam_cfg["ee_cam_orientation"]          # 例如 [0.0, 0.0, 0.0, 1.0]
    R_offset = np.array(p.getMatrixFromQuaternion(ee_cam_orn)).reshape(3, 3)

    # 计算摄像头在世界坐标系下的位置与旋转矩阵
    # 摄像头位置 = 末端执行器位置 + (末端旋转矩阵 * 摄像头 offset)
    cam_pos = ee_pos + R_ee @ ee_offset
    print("Camera position in world frame:", cam_pos)
    print("End effector position in world frame:", ee_pos)
    # 摄像头的旋转 = 末端旋转矩阵 * 摄像头相对于末端的旋转矩阵
    R_cam = R_ee @ R_offset

    # 将 extrinsics 封装成一个字典
    extrinsics = {"pos": cam_pos, "R": R_cam}

    # === 3. 利用采集的图像构建点云 ===
    # 注意：这里将 extrinsics 参数传递进去，使得 build_object_point_cloud 使用动态摄像头外参
    target_mask_ee = 5  # 根据实际情况调整
    try:
        pcd_ee = build_object_point_cloud(rgb_ee, depth_ee, seg_ee, target_mask_ee, config, extrinsics=extrinsics)
    except ValueError as e:
        print("Error building point cloud from end-effector camera:", e)
        pcd_ee = None
    
    # === 4. 在交互式窗口中显示点云 ===
    # 创建世界坐标系坐标轴
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    
    geometries = [coord_frame]
    if pcd_ee is not None:
        geometries.append(pcd_ee)
    
    print("Launching interactive Open3D visualization ...")
    o3d.visualization.draw_geometries(geometries)
    
    sim.close()

if __name__ == "__main__":
    with open("configs/test_config.yaml", "r") as stream:
        config = yaml.safe_load(stream)
    run_point_cloud_visualization(config)
