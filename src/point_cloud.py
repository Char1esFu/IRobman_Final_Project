import numpy as np
import open3d as o3d
import cv2

def build_object_point_cloud(rgb, depth, seg, target_mask_id, config, extrinsics=None):
    """
    利用 RGB、深度、seg 数据构建目标物体的点云，并转换到世界坐标系。
    
    参数:
      rgb: [H, W, 4] 的 RGBA 图像（颜色通道顺序为 R, G, B, A）
      depth: [H, W] 的深度图，深度缓冲值范围一般为 [0, 1]
      seg: [H, W] 的 segmentation mask（背景为 0，其它物体为非零 id）
      target_mask_id: 目标物体的 segmentation mask id
      config: 配置字典，其中必须包含如下键：
              world_settings -> camera -> width, height, fov, near, far, 等
      extrinsics: 可选，字典类型，包含:
              "pos": 3D 坐标（摄像头在世界坐标系下的位置）
              "R":   3x3 旋转矩阵（摄像头坐标系到世界坐标系的旋转）
              如果为 None，则使用 config 中的静态相机参数。
              
    返回:
      一个 open3d.geometry.PointCloud 对象，点云坐标均为世界坐标系下的值。
    """
    # 读取相机内参
    cam_cfg = config["world_settings"]["camera"]
    width = cam_cfg["width"]
    height = cam_cfg["height"]
    fov = cam_cfg["fov"]
    near = cam_cfg["near"]
    far = cam_cfg["far"]

    # 仅选取 segmentation mask 中目标物体对应的像素
    object_mask = (seg == target_mask_id)
    if np.count_nonzero(object_mask) == 0:
        raise ValueError(f"目标 mask id {target_mask_id} 未在 segmentation 中找到。")
    
    # 得到目标区域像素的行、列索引
    rows, cols = np.where(object_mask)
    
    # 从深度图中提取这些像素的深度值，并转换为实际距离（米）
    depth_buffer = depth[rows, cols]
    metric_depth = far * near / (far - (far - near) * depth_buffer)
    
    # 将像素坐标转换到归一化设备坐标（NDC）
    ndc_x = (2.0 * cols - width) / width
    ndc_y = -(2.0 * rows - height) / height  # 注意 y 轴取反
    
    aspect = width / height
    tan_half_fov = np.tan(np.deg2rad(fov / 2))
    
    # 计算相机坐标系下的点：x, y, z
    cam_x = ndc_x * aspect * tan_half_fov * metric_depth
    cam_y = ndc_y * tan_half_fov * metric_depth
    cam_z = metric_depth
    points_cam = np.vstack((cam_x, cam_y, cam_z))  # shape: (3, N)
    
    # 根据 extrinsics 参数决定使用哪套外参
    if extrinsics is None:
        # 使用静态相机参数
        cam_pos = np.array(cam_cfg["stat_cam_pos"])
        target_pos = np.array(cam_cfg["stat_cam_target_pos"])
        
        forward = target_pos - cam_pos
        forward = forward / np.linalg.norm(forward)
        # 这里假设 Z 轴为 up 方向，如果不满足可以根据需要调整
        right = np.cross(forward, np.array([0, 0, 1]))
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        R = np.column_stack([right, up, forward])  # shape: (3, 3)
    else:
        # 使用传入的外参
        cam_pos = np.array(extrinsics["pos"])
        R = np.array(extrinsics["R"])
    
    # 坐标转换：world_point = cam_pos + R @ cam_point
    points_world = cam_pos.reshape(3, 1) + R @ points_cam
    points_world = points_world.T  # shape: (N, 3)
    
    # 提取对应像素的颜色（RGB部分），归一化到 [0, 1]
    colors = rgb[rows, cols, :3].astype(np.float64) / 255.0
    
    # 构建 Open3D 点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_world)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd

def visualize_point_cloud_with_axes(pcd, axes_size=0.5):
    """
    交互式显示点云，并同时显示世界坐标系的三个轴。
    """
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axes_size, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, coord_frame])