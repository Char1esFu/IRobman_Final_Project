import os
os.environ["OPEN3D_DISABLE_WEB_VISUALIZER"] = "false"

import glob
import yaml
import time
import random
import numpy as np
import pybullet as p

from pybullet_object_models import ycb_objects  # type:ignore
from src.simulation import Simulation
from src.point_cloud import build_object_point_cloud, visualize_point_cloud_with_axes

def print_segmentation_info(seg):
    """
    打印 segmentation mask 中包含的所有唯一 id 信息，
    并尝试通过 p.getBodyInfo() 打印物体名称（如果可以）。
    
    注意：由于 segmentation mask 的像素值中可能包含了 link index，
    通常用如下方式提取 body id：
        body_id = seg_id & ((1 << 24) - 1)
    这里假定 PyBullet 使用低 24 位表示 body id。
    """
    unique_ids = np.unique(seg)
    print("Segmentation mask 中的唯一 id:")
    for seg_id in unique_ids:
        # 0 一般代表背景
        if seg_id == 0:
            print(f"  ID {seg_id}: 背景")
        else:
            # 提取 body id（低 24 位）
            body_id = int(seg_id) & ((1 << 24) - 1)
            try:
                body_info = p.getBodyInfo(body_id)
                # p.getBodyInfo 返回的是一个 tuple，第一个元素一般为物体名称（bytes类型）
                body_name = body_info[0].decode("utf-8") if body_info[0] is not None else "未知"
                print(f"  Segmentation ID {seg_id} -> body id {body_id}: 名称 = {body_name}")
            except Exception as e:
                print(f"  Segmentation ID {seg_id} -> body id {body_id}: 无法获取信息, 错误: {e}")


def run_point_cloud_visualization(config):
    print("Starting interactive point cloud visualization ...")
    
    # initialize simulation
    sim = Simulation(config)

    # get all object names from the YCB dataset
    object_root_path = ycb_objects.getDataPath()
    files = glob.glob(os.path.join(object_root_path, "Ycb*"))
    obj_names = [os.path.basename(file) for file in files]
    
    # randomly choose an object
    target_obj_name = random.choice(obj_names)
    print("Resetting simulation with random object:", target_obj_name)
    
    # reset simulation with the target object
    sim.reset(target_obj_name)
    # wait for the simulation to settle
    time.sleep(1)

    # get static and end-effector renders
    rgb, depth, seg = sim.get_static_renders()
    # rgb_ee, depth_ee, seg_ee = sim.get_ee_renders()
    
    # 打印 segmentation mask 信息
    print_segmentation_info(seg)
    
    # choose segmentation mask id for the target object
    target_mask_id = 5

    # build point cloud for the target object
    pcd = build_object_point_cloud(rgb, depth, seg, target_mask_id, config)
    # pcd_ee = build_object_point_cloud(rgb_ee, depth_ee, seg_ee, target_mask_id, config)
    
    # visualize the point cloud with axes
    visualize_point_cloud_with_axes(pcd, axes_size=0.5)
    # visualize_point_cloud_with_axes(pcd_ee, axes_size=0.5)
    sim.close()


if __name__ == "__main__":
    with open("configs/test_config.yaml", "r") as stream:
        config = yaml.safe_load(stream)
    run_point_cloud_visualization(config)
