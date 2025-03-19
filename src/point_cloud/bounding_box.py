import numpy as np
import pybullet as p
import open3d as o3d

class BoundingBox:
    """
    Class for calculating and visualizing point cloud bounding boxes
    """
    def __init__(self, point_cloud):
        """
        Initialize bounding box calculator
        
        Parameters:
        point_cloud: Open3D point cloud object or numpy point array (N,3)
        """
        # If input is an Open3D point cloud object, extract point coordinates
        if isinstance(point_cloud, o3d.geometry.PointCloud):
            self.points = np.asarray(point_cloud.points)
        else:
            self.points = np.asarray(point_cloud)
        
        # Initialize bounding box properties
        self.obb_corners = None    # Oriented bounding box vertices
        self.aabb_min = None       # Axis-aligned bounding box minimum point
        self.aabb_max = None       # Axis-aligned bounding box maximum point
        self.obb_dims = None       # Oriented bounding box dimensions
        self.rotation_matrix = None # Rotation matrix
        self.center = None         # Centroid
        self.height = None         # Object height
        self.debug_lines = []      # Line IDs for visualization
    
    def compute_obb(self):
        """
        Calculate the oriented bounding box (OBB) of the point cloud
        Implemented based on PCA analysis in the XY plane
        
        Returns:
        self: Returns self to support method chaining
        """
        # Check if point cloud is empty
        if len(self.points) == 0:
            raise ValueError("Point cloud is empty, cannot calculate bounding box")
        
        # Calculate point cloud centroid
        self.center = np.mean(self.points, axis=0)
        
        # 1. Project point cloud onto XY plane
        points_xy = self.points.copy()
        points_xy[:, 2] = 0  # Set Z coordinate to 0, projecting onto XY plane
        
        # 2. Perform PCA on the projected point cloud to find principal axes
        xy_mean = np.mean(points_xy, axis=0)
        xy_centered = points_xy - xy_mean
        cov_xy = np.cov(xy_centered.T)[:2, :2]  # Only take XY plane covariance
        eigenvalues, eigenvectors = np.linalg.eigh(cov_xy)
        
        # Sort eigenvalues and eigenvectors (descending)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # 3. Get principal axis directions, these are the rotation directions in the XY plane
        main_axis_x = np.array([eigenvectors[0, 0], eigenvectors[1, 0], 0])
        main_axis_y = np.array([eigenvectors[0, 1], eigenvectors[1, 1], 0])
        main_axis_z = np.array([0, 0, 1])  # Z axis remains vertical
        
        # Normalize main axes
        main_axis_x = main_axis_x / np.linalg.norm(main_axis_x)
        main_axis_y = main_axis_y / np.linalg.norm(main_axis_y)
        
        # 4. Build rotation matrix
        self.rotation_matrix = np.column_stack((main_axis_x, main_axis_y, main_axis_z))
        
        # 5. Rotate point cloud to new coordinate system
        points_rotated = np.dot(self.points - xy_mean, self.rotation_matrix)
        
        # 6. Calculate bounding box in new coordinate system
        min_point_rotated = np.min(points_rotated, axis=0)
        max_point_rotated = np.max(points_rotated, axis=0)
        
        # Calculate dimensions of rotated bounding box
        self.obb_dims = max_point_rotated - min_point_rotated
        self.height = self.obb_dims[2]
        
        # 7. Calculate the 8 vertices of the bounding box (in rotated coordinate system)
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
        
        # 8. Transform vertices back to original coordinate system
        self.obb_corners = np.dot(bbox_corners_rotated, self.rotation_matrix.T) + xy_mean
        
        # 9. Calculate axis-aligned bounding box (AABB) for grasp sampling (based on OBB vertices)
        self.aabb_min = np.min(self.obb_corners, axis=0)
        self.aabb_max = np.max(self.obb_corners, axis=0)
        
        return self
    
    def visualize_in_pybullet(self, color=(0, 1, 1), line_width=1, lifetime=0):
        """
        Visualize bounding box in PyBullet
        
        Parameters:
        color: Line color (R,G,B), default is cyan
        line_width: Line width, default is 1
        lifetime: Lifetime of lines (seconds), 0 means permanent
        
        Returns:
        debug_lines: List of line IDs for visualization
        """
        if self.obb_corners is None:
            raise ValueError("Please call compute_obb() first to calculate the bounding box")
        
        # Define the 12 edges of the bounding box
        bbox_lines = [
            # Bottom rectangle
            [self.obb_corners[0], self.obb_corners[1]],
            [self.obb_corners[1], self.obb_corners[2]],
            [self.obb_corners[2], self.obb_corners[3]],
            [self.obb_corners[3], self.obb_corners[0]],
            # Top rectangle
            [self.obb_corners[4], self.obb_corners[5]],
            [self.obb_corners[5], self.obb_corners[6]],
            [self.obb_corners[6], self.obb_corners[7]],
            [self.obb_corners[7], self.obb_corners[4]],
            # Connecting lines
            [self.obb_corners[0], self.obb_corners[4]],
            [self.obb_corners[1], self.obb_corners[5]],
            [self.obb_corners[2], self.obb_corners[6]],
            [self.obb_corners[3], self.obb_corners[7]]
        ]
        
        # Clear previous visualization lines
        self.clear_visualization()
        
        # Add new visualization lines
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
        Visualize the centroid of the bounding box in PyBullet
        
        Parameters:
        radius: Sphere radius, default is 0.01 meters
        color: Sphere color (R,G,B,A), default is red
        
        Returns:
        centroid_id: Object ID for visualization
        """
        if self.center is None:
            raise ValueError("Please call compute_obb() first to calculate the bounding box")
        
        # Create a sphere to represent the centroid
        visual_id = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=radius,
            rgbaColor=color
        )
        
        centroid_id = p.createMultiBody(
            baseMass=0,  # Mass of 0 indicates a static object
            baseVisualShapeIndex=visual_id,
            basePosition=self.center.tolist()
        )
        
        # Add text label
        p.addUserDebugText(
            f"Centroid ({self.center[0]:.3f}, {self.center[1]:.3f}, {self.center[2]:.3f})",
            self.center + np.array([0, 0, 0.05]),  # Display text 5cm above centroid
            [1, 1, 1],  # White text
            1.0  # Text size
        )
        
        return centroid_id
    
    def add_axes_visualization(self, length=0.1):
        """
        Visualize the principal axes of the bounding box in PyBullet
        
        Parameters:
        length: Axis length, default is 0.1 meters
        
        Returns:
        axis_lines: List of line IDs for visualization
        """
        if self.rotation_matrix is None or self.center is None:
            raise ValueError("Please call compute_obb() first to calculate the bounding box")
        
        # Get principal axis directions
        axis_x = self.rotation_matrix[:, 0] * length
        axis_y = self.rotation_matrix[:, 1] * length
        axis_z = self.rotation_matrix[:, 2] * length
        
        # Add principal axes visualization
        axis_lines = []
        
        # X axis - red
        line_id = p.addUserDebugLine(
            self.center,
            self.center + axis_x,
            [1, 0, 0],
            3,
            0
        )
        axis_lines.append(line_id)
        
        # Y axis - green
        line_id = p.addUserDebugLine(
            self.center,
            self.center + axis_y,
            [0, 1, 0],
            3,
            0
        )
        axis_lines.append(line_id)
        
        # Z axis - blue
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
        Clear visualization lines in PyBullet
        """
        for line_id in self.debug_lines:
            p.removeUserDebugItem(line_id)
        
        self.debug_lines = []
    
    def get_dimensions(self):
        """
        Get the dimensions of the bounding box
        
        Returns:
        dimensions: Bounding box dimensions [length, width, height]
        """
        if self.obb_dims is None:
            raise ValueError("Please call compute_obb() first to calculate the bounding box")
        
        return self.obb_dims
    
    def get_height(self):
        """
        Get object height
        
        Returns:
        height: Object height
        """
        if self.height is None:
            raise ValueError("Please call compute_obb() first to calculate the bounding box")
        
        return self.height
    
    def get_center(self):
        """
        Get the center point of the bounding box
        
        Returns:
        center: Center point of the bounding box [x, y, z]
        """
        if self.center is None:
            raise ValueError("Please call compute_obb() first to calculate the bounding box")
        
        return self.center
    
    def get_rotation_matrix(self):
        """
        Get the rotation matrix of the bounding box
        
        Returns:
        rotation_matrix: 3x3 rotation matrix
        """
        if self.rotation_matrix is None:
            raise ValueError("Please call compute_obb() first to calculate the bounding box")
        
        return self.rotation_matrix
    
    def get_corners(self):
        """
        Get the 8 vertices of the bounding box
        
        Returns:
        corners: Coordinates of 8 vertices, shape (8,3)
        """
        if self.obb_corners is None:
            raise ValueError("Please call compute_obb() first to calculate the bounding box")
        
        return self.obb_corners
    
    def get_aabb(self):
        """
        Get the minimum and maximum points of the axis-aligned bounding box
        
        Returns:
        (min_point, max_point): Minimum and maximum points of the bounding box
        """
        if self.aabb_min is None or self.aabb_max is None:
            raise ValueError("Please call compute_obb() first to calculate the bounding box")
        
        return self.aabb_min, self.aabb_max
    
    @staticmethod
    def compute_point_cloud_bbox(sim, collector, point_clouds, visualize_cloud=True):
        """
        Calculate and visualize point cloud bounding box
        
        Parameters:
        sim: Simulation environment object
        collector: Point cloud collector object
        point_clouds: Collected point cloud data
        visualize_cloud: Whether to visualize the point cloud
        
        Returns:
        bbox: Calculated bounding box object
        """
        print("\nStep 2: Calculate and visualize bounding box...")
        
        # Visualize collected point clouds
        if visualize_cloud and point_clouds:
            # Display individual point clouds
            print("\nVisualizing individual point clouds...")
            collector.visualize_point_clouds(point_clouds, show_merged=False)
        
        # Merge point clouds
        print("\nMerging point clouds...")
        merged_cloud = collector.merge_point_clouds(point_clouds)
        
        # Visualize merged point cloud
        if visualize_cloud and merged_cloud is not None:
            print("\nVisualizing merged point cloud...")
            # Create a list containing only the merged point cloud
            merged_cloud_data = [{
                'point_cloud': merged_cloud,
                'camera_position': np.array([0, 0, 0]),  # Placeholder
                'camera_rotation': np.eye(3)  # Placeholder
            }]
            collector.visualize_point_clouds(merged_cloud_data, show_merged=False)
        
        # Calculate bounding box
        print("\nCalculating bounding box...")
        bbox = BoundingBox(merged_cloud)
        bbox.compute_obb()
        
        # Visualize bounding box
        print("\nVisualizing bounding box...")
        bbox.visualize_in_pybullet(color=(0, 1, 1), line_width=3)
        
        # Visualize centroid
        centroid_id = bbox.add_centroid_visualization(radius=0.02)
        
        # Visualize principal axes
        axis_lines = bbox.add_axes_visualization(length=0.15)
        
        # Print bounding box information
        print(f"\nBounding box information:")
        print(f"Object height: {bbox.get_height():.4f} meters")
        print(f"Bounding box dimensions: {bbox.get_dimensions()}")
        center = bbox.get_center()
        print(f"Centroid coordinates: ({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f})")
        
        return bbox
