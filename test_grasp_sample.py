import open3d as o3d
import numpy as np
import re

def read_grasp_centers(file_path):
    """
    从文件中读取grasp_center点的坐标
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 使用正则表达式提取坐标
    pattern = r'grasp_center: \[(.*?)\]'
    matches = re.findall(pattern, content)
    
    points = []
    for match in matches:
        # 将匹配到的字符串转换为坐标
        coords = match.strip().split()
        # 处理科学计数法和普通数字格式
        point = [float(coord) for coord in coords]
        points.append(point)
    
    return np.array(points)

def create_point_cloud(points):
    """
    创建Open3D点云对象
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 为点云设置颜色 - 使用红色
    colors = [[1, 0, 0] for _ in range(len(points))]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd

def visualize_points(pcd):
    """
    可视化点云
    """
    # 创建坐标系，以便更好地理解空间位置
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.1, origin=[0, 0, 0])
    
    # 创建一些额外的参考点来增强视觉效果
    reference_points = []
    # 原点
    reference_points.append(o3d.geometry.TriangleMesh.create_sphere(radius=0.01))
    reference_points[-1].paint_uniform_color([0, 1, 0])  # 绿色原点
    
    # 计算点云的中心点
    center = np.mean(np.asarray(pcd.points), axis=0)
    center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.002)
    center_sphere.translate(center)
    center_sphere.paint_uniform_color([0, 0, 1])  # 蓝色中心点
    reference_points.append(center_sphere)
    
    # 创建自定义可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Grasp Centers Visualization")
    
    # 添加点云和参考对象
    vis.add_geometry(pcd)
    vis.add_geometry(coordinate_frame)
    for ref_point in reference_points:
        vis.add_geometry(ref_point)
    
    # 设置视角
    opt = vis.get_render_option()
    opt.background_color = np.array([0.8, 0.8, 0.8])  # 浅灰色背景
    opt.point_size = 5.0  # 增大点的尺寸，更容易看清
    
    # 更新可视化窗口并显示
    vis.update_renderer()
    vis.poll_events()
    vis.update_geometry(pcd)
    
    # 调整相机位置
    ctr = vis.get_view_control()
    ctr.set_lookat(center)
    ctr.set_up([0, 0, 1])  # Z轴向上
    ctr.set_front([1, 0, 0])  # 看向X轴正方向
    ctr.set_zoom(0.8)
    
    print("按住鼠标左键旋转视图，滚动鼠标滚轮缩放，按住鼠标右键平移视图")
    print("按'Q'或关闭窗口退出程序")
    
    # 运行可视化循环
    vis.run()
    vis.destroy_window()

def main():
    file_path = "paste.txt"  # 你的文本文件路径
    
    # 读取点数据
    points = read_grasp_centers(file_path)
    print(f"读取了 {len(points)} 个点")
    
    # 创建点云
    pcd = create_point_cloud(points)
    
    # 计算点云的统计信息
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    mean_coords = np.mean(points, axis=0)
    std_coords = np.std(points, axis=0)
    
    print("\n点云统计信息:")
    print(f"点的数量: {len(points)}")
    print(f"最小坐标: {min_coords}")
    print(f"最大坐标: {max_coords}")
    print(f"平均坐标: {mean_coords}")
    print(f"坐标标准差: {std_coords}")
    print(f"坐标范围: {max_coords - min_coords}")
    
    # 可视化点云
    visualize_points(pcd)

if __name__ == "__main__":
    main()