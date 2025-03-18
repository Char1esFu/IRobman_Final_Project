import numpy as np
import pybullet as p
import cv2

class ObstacleTracker:
    def __init__(self, n_obstacles=2, exp_settings=None):
        """
        Args:
            n_obstacles: Number of obstacles to track
            config: Configuration dictionary from yaml
        """
        if exp_settings is None:
            raise ValueError("Config cannot be loaded")

        self.camera_settings = exp_settings["world_settings"]["camera"]
        self.n_obstacles = n_obstacles
        # Store latest measurements
        self.latest_positions = np.zeros((n_obstacles, 3))
        self.latest_radius = [0.0 for _ in range(n_obstacles)]
        

    def convert_depth_to_meters(self, depth_buffer):
        """Convert depth buffer to metric depth."""
        far = self.camera_settings["far"]
        near = self.camera_settings["near"]

        return far * near / (far - (far - near) * depth_buffer)    
    
    def pixel_to_world(self, pixel_x, pixel_y, depth, radius=0):
        """
        Convert pixel coordinates to world coordinates, adjusting for sphere center.
        
        Args:
            pixel_x: x coordinate in image space
            pixel_y: y coordinate in image space
            depth: depth value from depth buffer
            radius: radius of the sphere (to adjust from surface to center)
        
        Returns:
            world_point: 3D coordinates in world space
        """
        width = self.camera_settings["width"]
        height = self.camera_settings["height"]
        fov = self.camera_settings["fov"] # conventionally defined in the direction of height
        
        # 1. Get camera parameters
        cam_pos = np.array(self.camera_settings["stat_cam_pos"])
        target_pos = np.array(self.camera_settings["stat_cam_target_pos"])
        
        # 2. Pixel to NDC (Normalized Device Coordinates)
        ndc_x = (2.0 * pixel_x - width) / width
        ndc_y = -(2.0 * pixel_y - height) / height
        
        # 3. NDC to camera space
        aspect = width / height
        tan_half_fov = np.tan(np.deg2rad(fov / 2))
        cam_x = ndc_x * aspect * tan_half_fov * depth
        cam_y = ndc_y * tan_half_fov * depth
        cam_z = depth
        
        # 4. Create camera space basis
        forward = target_pos - cam_pos
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, np.array([0, 0, 1]))
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        
        # 5. Camera space to world space
        R = np.column_stack([right, up, forward])
        cam_point = np.array([cam_x, cam_y, cam_z])
        surface_point = cam_pos + R @ cam_point
        
        if radius > 0:
            # Calculate direction from camera to surface point
            direction = surface_point - cam_pos
            direction = direction / np.linalg.norm(direction)
            
            # Offset the surface point by radius along this direction
            center_point = surface_point + direction * radius
            return center_point
        else:
            return surface_point
    
    def calculate_metric_radius(self, area, depth):
        """Calculate sphere radius with corrective factor"""
        height = self.camera_settings["height"]
        fov_rad = np.deg2rad(self.camera_settings["fov"])
        
        # projected pixel radius
        pixel_radius = np.sqrt(area / np.pi)
        
        # calculate base radius
        base_radius = (pixel_radius * depth * 2 * np.tan(fov_rad/2)) / height
        
        
        return base_radius
    
    def detect_obstacles(self, rgb, depth, seg):
        """Detect obstacles using fixed segmentation mask IDs (6 and 7)"""
        detections = []
        potential_balls = []
        
        # 固定的障碍物ID
        obstacle_ids = [6, 7]
        # print(f"\n使用固定的障碍物ID: {obstacle_ids}")
        
        # 直接处理固定ID的障碍物
        for obj_id in obstacle_ids:
            # 检查该ID是否存在于分割掩码中
            if obj_id not in np.unique(seg):
                print(f"警告: ID {obj_id} 不在当前分割掩码中")
                continue
                
            mask = (seg == obj_id).astype(np.uint8)
            
            # 找到轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                print(f"警告: ID {obj_id} 没有找到有效轮廓")
                continue
                
            # 使用最大的轮廓
            contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(contour)
            
            # 计算轮廓中心
            M = cv2.moments(contour)
            if M['m00'] == 0:
                print(f"警告: ID {obj_id} 的轮廓面积为零")
                continue
                
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            
            # 检查深度
            depth_buffer = depth[cy, cx]
            metric_depth = self.convert_depth_to_meters(depth_buffer)
            
            # 计算半径
            base_radius = self.calculate_metric_radius(area, metric_depth)
            
            # 计算球心的世界坐标
            world_pos = self.pixel_to_world(cx, cy, metric_depth, radius=base_radius)
            
            # 添加到潜在球体列表
            potential_balls.append({
                'id': obj_id,
                'center': (cx, cy),
                'world_pos': world_pos,
                'depth': metric_depth,
                'area': area,
                'radius': base_radius
            })

        # 直接使用所有检测到的球体
        for ball in potential_balls:
            detections.append(np.array([
                ball['world_pos'][0], 
                ball['world_pos'][1], 
                ball['world_pos'][2], 
                ball['radius']
            ]))

        return detections
    
    def update(self, detections):
        """Update tracking with new detections."""
        for i, detection in enumerate(detections):
            if i < self.n_obstacles:
                self.latest_positions[i] = detection[:3]  # set initial position
                self.latest_radius[i] = detection[3]     # store radius separately
        return self.latest_positions

          
    # 3d bounding box
    def visualize_tracking_3d(self, tracked_positions):
        """Visualize tracking boxes in 3D space"""
        debug_ids = []
        
        for i, pos in enumerate(tracked_positions):
            # access the estimated radius
            half_size = self.latest_radius[i]
            
            # 8 corners of the bounding box
            corners = [
                [pos[0]-half_size, pos[1]-half_size, pos[2]-half_size],
                [pos[0]+half_size, pos[1]-half_size, pos[2]-half_size],
                [pos[0]+half_size, pos[1]+half_size, pos[2]-half_size],
                [pos[0]-half_size, pos[1]+half_size, pos[2]-half_size],
                [pos[0]-half_size, pos[1]-half_size, pos[2]+half_size],
                [pos[0]+half_size, pos[1]-half_size, pos[2]+half_size],
                [pos[0]+half_size, pos[1]+half_size, pos[2]+half_size],
                [pos[0]-half_size, pos[1]+half_size, pos[2]+half_size]
            ]
            
            # 4 bottom edges
            debug_ids.append(p.addUserDebugLine(corners[0], corners[1], [0, 1, 0]))
            debug_ids.append(p.addUserDebugLine(corners[1], corners[2], [0, 1, 0]))
            debug_ids.append(p.addUserDebugLine(corners[2], corners[3], [0, 1, 0]))
            debug_ids.append(p.addUserDebugLine(corners[3], corners[0], [0, 1, 0]))
            
            # 4 top edges
            debug_ids.append(p.addUserDebugLine(corners[4], corners[5], [0, 1, 0]))
            debug_ids.append(p.addUserDebugLine(corners[5], corners[6], [0, 1, 0]))
            debug_ids.append(p.addUserDebugLine(corners[6], corners[7], [0, 1, 0]))
            debug_ids.append(p.addUserDebugLine(corners[7], corners[4], [0, 1, 0]))
            
            # 4 vertical edges
            debug_ids.append(p.addUserDebugLine(corners[0], corners[4], [0, 1, 0]))
            debug_ids.append(p.addUserDebugLine(corners[1], corners[5], [0, 1, 0]))
            debug_ids.append(p.addUserDebugLine(corners[2], corners[6], [0, 1, 0]))
            debug_ids.append(p.addUserDebugLine(corners[3], corners[7], [0, 1, 0]))
        
        return debug_ids
    
    def get_obstacle_state(self, obstacle_index):
        """Get full state estimate for an obstacle.
        
        Args:
            obstacle_index: Index of the obstacle (0 or 1)
            
        Returns:
            dict containing:
                position: np.array([x, y, z])
                velocity: np.array([vx, vy, vz])
                radius: float
        """
        if not (0 <= obstacle_index < self.n_obstacles):
            raise ValueError(f"Invalid obstacle index: {obstacle_index}")

        state = {
            'position': self.latest_positions[obstacle_index], 
            'radius': self.latest_radius[obstacle_index]
        }
        return state
        
    def get_all_obstacle_states(self):
        """Get state estimates for all obstacles.
        
        Returns:
            List of obstacle state dictionaries
        """
        return [self.get_obstacle_state(i) for i in range(self.n_obstacles)]