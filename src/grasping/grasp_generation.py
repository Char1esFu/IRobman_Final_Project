import numpy as np
from typing import Tuple, Sequence, Optional, Any
import open3d as o3d


class GraspGeneration:
    def __init__(self):
        pass

    def sample_grasps(
        self,
        center_point: np.ndarray,
        num_grasps: int,
        offset: float = 0.1,
    ) -> Sequence[Tuple[np.ndarray, np.ndarray]]:
        """
        Generates multiple random grasp poses around a given point cloud.
        Uses PyBullet's coordinate system convention where:
        - roll=0 means gripper pointing up (+Z)
        - roll=180 means gripper pointing down (-Z)
        - pitch and yaw define the tilting from vertical axis

        Args:
            center: Center around which to sample grasps.
            num_grasps: Number of random grasp poses to generate
            offset: Maximum distance offset from the center (meters)

        Returns:
            list: List of rotations and Translations
        """

        grasp_poses_list = []
        for idx in range(num_grasps):
            # Sample position offset
            theta = np.random.uniform(0, 2*np.pi)  # Rotation around vertical axis
            r = np.random.uniform(0, offset)
            
            # Calculate position with offset mainly in x-y plane
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            z = np.random.uniform(-offset/4, offset/4)  # Smaller vertical offset
            grasp_center = center_point + np.array([x, y, z])

            # Generate rotation for downward-facing grasp
            # Roll: around x-axis, 180° (pointing down)
            roll = np.radians(180)
            # Pitch: around y-axis, small variation for tilting
            pitch = np.radians(0)
            # Yaw: around z-axis, full rotation allowed
            yaw = np.random.uniform(-np.pi, np.pi)

            # Convert Euler angles to rotation matrices
            Rx = np.array([
                [1, 0, 0],
                [0, np.cos(roll), -np.sin(roll)],
                [0, np.sin(roll), np.cos(roll)]
            ])
            
            Ry = np.array([
                [np.cos(pitch), 0, np.sin(pitch)],
                [0, 1, 0],
                [-np.sin(pitch), 0, np.cos(pitch)]
            ])
            
            Rz = np.array([
                [np.cos(yaw), -np.sin(yaw), 0],
                [np.sin(yaw), np.cos(yaw), 0],
                [0, 0, 1]
            ])
            
            # Combine rotations: R = Rz @ Ry @ Rx
            R = Rz @ Ry @ Rx

            assert R.shape == (3, 3)
            assert grasp_center.shape == (3,)
            grasp_poses_list.append((R, grasp_center))

        return grasp_poses_list
    

    def check_grasp_collision(
        self,
        grasp_meshes: Sequence[o3d.geometry.TriangleMesh],
        # object_mesh: o3d.geometry.TriangleMesh,
        object_pcd,
        num_colisions: int = 10,
        tolerance: float = 0.00001) -> bool:
        """
        Checks for collisions between a gripper grasp pose and target object
        using point cloud sampling.

        Args:
            grasp_meshes: List of mesh geometries representing the gripper components
            object_mesh: Triangle mesh of the target object
            num_collisions: Threshold on how many points to check
            tolerance: Distance threshold for considering a collision (in meters)

        Returns:
            bool: True if collision detected between gripper and object, False otherwise
        """
        # Combine gripper meshes
        combined_gripper = o3d.geometry.TriangleMesh()
        for mesh in grasp_meshes:
            combined_gripper += mesh

        # Sample points from both meshes
        num_points = 5000 # Subsample both meshes to this many points
        gripper_pcl = combined_gripper.sample_points_uniformly(number_of_points=num_points)
        # object_pcl = object_mesh.sample_points_uniformly(number_of_points=num_points)
        object_pcl = object_pcd

        # Build KDTree for object points
        is_collision = False
        object_kd_tree = o3d.geometry.KDTreeFlann(object_pcl)
        collision_count = 0
        for point in gripper_pcl.points:
            [_, idx, distance] = object_kd_tree.search_knn_vector_3d(point, 1)
            if distance[0] < tolerance:
                collision_count += 1
                if collision_count >= num_colisions:
                    return True  # Collision detected

        return is_collision
    
    def grasp_dist_filter(self,
                        center_grasp: np.ndarray,
                        mesh_center: np.ndarray,
                        tolerance: float = 0.05)->bool:
        is_within_range = False
        #######################TODO#######################
        if np.linalg.norm(center_grasp - mesh_center) < tolerance:
            is_within_range = True
        ##################################################
        return is_within_range
    

    def check_grasp_containment(
        self,
        left_finger_center: np.ndarray,
        right_finger_center: np.ndarray,
        finger_length: float,
        object_pcd: o3d.geometry.PointCloud,
        num_rays: int,
        rotation_matrix: np.ndarray, # rotation-mat
    ) -> Tuple[bool, float]:
        """
        Checks if any line between the gripper fingers intersects with the object mesh.
        Evaluates grasp quality based on:
        1. Number of rays that hit the object
        2. Average penetration depth of the rays
        3. Position of hits along the finger length

        Args:
            left_finger_center: Center of Left finger of grasp
            right_finger_center: Center of Right finger of grasp
            finger_length: Finger Length of the gripper
            object_pcd: Point Cloud of the target object
            num_rays: Number of rays to cast between fingers
            rotation_matrix: Rotation matrix for the grasp

        Returns:
            tuple[bool, float]: (intersection_exists, grasp_quality)
            - intersection_exists: True if valid grasp found
            - grasp_quality: Quality score combining hit ratio and depth
        """
        left_center = np.asarray(left_finger_center)
        right_center = np.asarray(right_finger_center)

        # Create mesh for ray casting
        obj_triangle_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            pcd=object_pcd, alpha=0.016)
        obj_triangle_mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(obj_triangle_mesh)
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(obj_triangle_mesh_t)

        # Calculate grasp parameters
        hand_width = np.linalg.norm(left_center-right_center)
        finger_vec = np.array([0, finger_length, 0])
        ray_direction = (left_center - right_center)/hand_width

        # Move right finger to start position
        right_center = right_center - rotation_matrix.dot(finger_vec/2)
        
        # Cast rays along finger length
        rays = []
        for i in range(num_rays):
            right_new_center = right_center + rotation_matrix.dot((i/num_rays)*finger_vec)
            rays.append([np.concatenate([right_new_center, ray_direction])])

        rays_t = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
        ans = scene.cast_rays(rays_t)

        # Analyze ray hits
        rays_hit = 0
        total_depth = 0
        min_depth_threshold = 0.01  # Minimum penetration depth (1cm)
        max_depth_threshold = hand_width * 0.8  # Maximum penetration depth (80% of hand width)
        
        # Weight hits based on position along finger
        position_weights = np.linspace(0.5, 1.0, num_rays)  # Higher weight for hits near finger tip
        
        for idx, hit_point in enumerate(ans['t_hit']):
            hit_depth = hit_point[0]
            if min_depth_threshold < hit_depth < max_depth_threshold:
                rays_hit += 1
                # Weight the depth by position along finger
                total_depth += hit_depth * position_weights[idx]

        # Calculate grasp quality metrics
        hit_ratio = rays_hit / num_rays
        avg_depth_ratio = total_depth / (rays_hit * hand_width) if rays_hit > 0 else 0
        
        # Combine metrics into final quality score
        # Hit ratio and depth ratio are equally weighted
        grasp_quality = 0.5 * hit_ratio + 0.5 * avg_depth_ratio if rays_hit > 0 else 0
        
        return rays_hit > 0, grasp_quality
