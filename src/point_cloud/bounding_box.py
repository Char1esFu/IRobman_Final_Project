import numpy as np
import pybullet as p
import open3d as o3d

class BoundingBox:
    """
    用于计算和可视化点云边界框的类
    """
    def __init__(self, point_cloud):
        """
        初始化边界框计算器
        
        参数:
        point_cloud: Open3D点云对象或numpy点数组(N,3)
        """
        # 如果输入是Open3D点云对象，提取点坐标
        if isinstance(point_cloud, o3d.geometry.PointCloud):
            self.points = np.asarray(point_cloud.points)
        else:
            self.points = np.asarray(point_cloud)
        
        # 初始化边界框属性
        self.obb_corners = None    # 旋转边界框的顶点
        self.aabb_min = None       # 轴对齐边界框最小点
        self.aabb_max = None       # 轴对齐边界框最大点
        self.obb_dims = None       # 旋转边界框的尺寸
        self.rotation_matrix = None # 旋转矩阵
        self.center = None         # 质心
        self.height = None         # 物体高度
        self.debug_lines = []      # 用于可视化的线条ID
    
    def compute_obb(self):
        """
        计算点云的旋转边界框(OBB)
        基于XY平面的PCA分析实现
        
        返回:
        self: 返回自身以支持方法链式调用
        """
        # 检查点云是否为空
        if len(self.points) == 0:
            raise ValueError("点云为空，无法计算边界框")
        
        # 计算点云质心
        self.center = np.mean(self.points, axis=0)
        
        # 1. 将点云投影到XY平面
        points_xy = self.points.copy()
        points_xy[:, 2] = 0  # 将Z坐标设为0，投影到XY平面
        
        # 2. 对XY平面上的点云进行PCA，找到主轴方向
        xy_mean = np.mean(points_xy, axis=0)
        xy_centered = points_xy - xy_mean
        cov_xy = np.cov(xy_centered.T)[:2, :2]  # 只取XY平面的协方差
        eigenvalues, eigenvectors = np.linalg.eigh(cov_xy)
        
        # 排序特征值和特征向量（降序）
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # 3. 获取主轴方向，这些是XY平面内的旋转方向
        main_axis_x = np.array([eigenvectors[0, 0], eigenvectors[1, 0], 0])
        main_axis_y = np.array([eigenvectors[0, 1], eigenvectors[1, 1], 0])
        main_axis_z = np.array([0, 0, 1])  # Z轴保持垂直
        
        # 归一化主轴
        main_axis_x = main_axis_x / np.linalg.norm(main_axis_x)
        main_axis_y = main_axis_y / np.linalg.norm(main_axis_y)
        
        # 4. 构建旋转矩阵
        self.rotation_matrix = np.column_stack((main_axis_x, main_axis_y, main_axis_z))
        
        # 5. 将点云旋转到新坐标系
        points_rotated = np.dot(self.points - xy_mean, self.rotation_matrix)
        
        # 6. 在新坐标系中计算边界框
        min_point_rotated = np.min(points_rotated, axis=0)
        max_point_rotated = np.max(points_rotated, axis=0)
        
        # 计算旋转后的边界框尺寸
        self.obb_dims = max_point_rotated - min_point_rotated
        self.height = self.obb_dims[2]
        
        # 7. 计算边界框的8个顶点（在旋转坐标系中）
        bbox_corners_rotated = np.array([
            [min_point_rotated[0], min_point_rotated[1], min_point_rotated[2]],
            [max_point_rotated[0], min_point_rotated[1], min_point_rotated[2]],
            [max_point_rotated[0], max_point_rotated[1], min_point_rotated[2]],
            [min_point_rotated[0], max_point_rotated[1], min_point_rotated[2]],
            [min_point_rotated[0], min_point_rotated[1], max_point_rotated[2]],
            [max_point_rotated[0], min_point_rotated[1], max_point_rotated[2]],
            [max_point_rotated[0], max_point_rotated[1], max_point_rotated[2]],
            [min_point_rotated[0], max_point_rotated[1], max_point_rotated[2]],
        ])
        
        # 8. 将顶点变换回原始坐标系
        self.obb_corners = np.dot(bbox_corners_rotated, self.rotation_matrix.T) + xy_mean
        
        # 9. 计算轴对齐边界框(AABB)用于抓取采样（基于OBB的顶点）
        self.aabb_min = np.min(self.obb_corners, axis=0)
        self.aabb_max = np.max(self.obb_corners, axis=0)
        
        return self
    
    def visualize_in_pybullet(self, color=(0, 1, 1), line_width=1, lifetime=0):
        """
        在PyBullet中可视化边界框
        
        参数:
        color: 线条颜色(R,G,B)，默认为青色
        line_width: 线条宽度，默认为1
        lifetime: 线条的生命周期（秒），0表示永久存在
        
        返回:
        debug_lines: 用于可视化的线条ID列表
        """
        if self.obb_corners is None:
            raise ValueError("请先调用compute_obb()计算边界框")
        
        # 定义边界框的12条边
        bbox_lines = [
            # 底部矩形
            [self.obb_corners[0], self.obb_corners[1]],
            [self.obb_corners[1], self.obb_corners[2]],
            [self.obb_corners[2], self.obb_corners[3]],
            [self.obb_corners[3], self.obb_corners[0]],
            # 顶部矩形
            [self.obb_corners[4], self.obb_corners[5]],
            [self.obb_corners[5], self.obb_corners[6]],
            [self.obb_corners[6], self.obb_corners[7]],
            [self.obb_corners[7], self.obb_corners[4]],
            # 连接线
            [self.obb_corners[0], self.obb_corners[4]],
            [self.obb_corners[1], self.obb_corners[5]],
            [self.obb_corners[2], self.obb_corners[6]],
            [self.obb_corners[3], self.obb_corners[7]]
        ]
        
        # 清除之前的可视化线条
        self.clear_visualization()
        
        # 添加新的可视化线条
        for line in bbox_lines:
            line_id = p.addUserDebugLine(
                line[0], 
                line[1], 
                color,
                line_width, 
                lifetime
            )
            self.debug_lines.append(line_id)
        
        return self.debug_lines
    
    def add_centroid_visualization(self, radius=0.01, color=(1, 0, 0, 1)):
        """
        在PyBullet中可视化边界框的质心
        
        参数:
        radius: 球体半径，默认为0.01米
        color: 球体颜色(R,G,B,A)，默认为红色
        
        返回:
        centroid_id: 用于可视化的物体ID
        """
        if self.center is None:
            raise ValueError("请先调用compute_obb()计算边界框")
        
        # 创建一个球体表示质心
        visual_id = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=radius,
            rgbaColor=color
        )
        
        centroid_id = p.createMultiBody(
            baseMass=0,  # 质量为0表示静态物体
            baseVisualShapeIndex=visual_id,
            basePosition=self.center.tolist()
        )
        
        # 添加文本标签
        p.addUserDebugText(
            f"Centroid ({self.center[0]:.3f}, {self.center[1]:.3f}, {self.center[2]:.3f})",
            self.center + np.array([0, 0, 0.05]),  # 在质心上方5cm处显示文本
            [1, 1, 1],  # 白色文本
            1.0  # 文本大小
        )
        
        return centroid_id
    
    def add_axes_visualization(self, length=0.1):
        """
        在PyBullet中可视化边界框的主轴
        
        参数:
        length: 坐标轴长度，默认为0.1米
        
        返回:
        axis_lines: 用于可视化的线条ID列表
        """
        if self.rotation_matrix is None or self.center is None:
            raise ValueError("请先调用compute_obb()计算边界框")
        
        # 获取主轴方向
        axis_x = self.rotation_matrix[:, 0] * length
        axis_y = self.rotation_matrix[:, 1] * length
        axis_z = self.rotation_matrix[:, 2] * length
        
        # 添加主轴可视化
        axis_lines = []
        
        # X轴 - 红色
        line_id = p.addUserDebugLine(
            self.center,
            self.center + axis_x,
            [1, 0, 0],
            3,
            0
        )
        axis_lines.append(line_id)
        
        # Y轴 - 绿色
        line_id = p.addUserDebugLine(
            self.center,
            self.center + axis_y,
            [0, 1, 0],
            3,
            0
        )
        axis_lines.append(line_id)
        
        # Z轴 - 蓝色
        line_id = p.addUserDebugLine(
            self.center,
            self.center + axis_z,
            [0, 0, 1],
            3,
            0
        )
        axis_lines.append(line_id)
        
        return axis_lines
    
    def clear_visualization(self):
        """
        清除PyBullet中的可视化线条
        """
        for line_id in self.debug_lines:
            p.removeUserDebugItem(line_id)
        
        self.debug_lines = []
    
    def get_dimensions(self):
        """
        获取边界框的尺寸
        
        返回:
        dimensions: 边界框的尺寸[长, 宽, 高]
        """
        if self.obb_dims is None:
            raise ValueError("请先调用compute_obb()计算边界框")
        
        return self.obb_dims
    
    def get_height(self):
        """
        获取物体高度
        
        返回:
        height: 物体高度
        """
        if self.height is None:
            raise ValueError("请先调用compute_obb()计算边界框")
        
        return self.height
    
    def get_center(self):
        """
        获取边界框的中心点
        
        返回:
        center: 边界框的中心点[x, y, z]
        """
        if self.center is None:
            raise ValueError("请先调用compute_obb()计算边界框")
        
        return self.center
    
    def get_rotation_matrix(self):
        """
        获取边界框的旋转矩阵
        
        返回:
        rotation_matrix: 3x3旋转矩阵
        """
        if self.rotation_matrix is None:
            raise ValueError("请先调用compute_obb()计算边界框")
        
        return self.rotation_matrix
    
    def get_corners(self):
        """
        获取边界框的8个顶点
        
        返回:
        corners: 8个顶点的坐标，形状为(8,3)
        """
        if self.obb_corners is None:
            raise ValueError("请先调用compute_obb()计算边界框")
        
        return self.obb_corners
    
    def get_aabb(self):
        """
        获取轴对齐边界框的最小点和最大点
        
        返回:
        (min_point, max_point): 边界框的最小点和最大点
        """
        if self.aabb_min is None or self.aabb_max is None:
            raise ValueError("请先调用compute_obb()计算边界框")
        
        return self.aabb_min, self.aabb_max
    
    @staticmethod
    def compute_point_cloud_bbox(sim, collector, point_clouds, visualize_cloud=True):
        """
        计算和可视化点云边界框
        
        参数:
        sim: 仿真环境对象
        collector: 点云收集器对象
        point_clouds: 收集的点云数据
        visualize_cloud: 是否可视化点云
        
        返回:
        bbox: 计算的边界框对象
        """
        print("\n步骤2: 计算和可视化边界框...")
        
        # 可视化收集的点云
        if visualize_cloud and point_clouds:
            # 显示单独的点云
            print("\n可视化单独点云...")
            collector.visualize_point_clouds(point_clouds, show_merged=False)
        
        # 合并点云
        print("\n合并点云...")
        merged_cloud = collector.merge_point_clouds(point_clouds)
        
        # 可视化合并点云
        if visualize_cloud and merged_cloud is not None:
            print("\n可视化合并点云...")
            # 创建一个只包含合并点云的列表
            merged_cloud_data = [{
                'point_cloud': merged_cloud,
                'camera_position': np.array([0, 0, 0]),  # 占位符
                'camera_rotation': np.eye(3)  # 占位符
            }]
            collector.visualize_point_clouds(merged_cloud_data, show_merged=False)
        
        # 计算边界框
        print("\n计算边界框...")
        bbox = BoundingBox(merged_cloud)
        bbox.compute_obb()
        
        # 可视化边界框
        print("\n可视化边界框...")
        bbox.visualize_in_pybullet(color=(0, 1, 1), line_width=3)
        
        # 可视化中心点
        centroid_id = bbox.add_centroid_visualization(radius=0.02)
        
        # 可视化主轴
        axis_lines = bbox.add_axes_visualization(length=0.15)
        
        # 打印边界框信息
        print(f"\n边界框信息:")
        print(f"物体高度: {bbox.get_height():.4f}米")
        print(f"边界框尺寸: {bbox.get_dimensions()}")
        center = bbox.get_center()
        print(f"质心坐标: ({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f})")
        
        return bbox
