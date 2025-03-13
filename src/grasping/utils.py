import numpy as np
import open3d as o3d
from typing import Any, Sequence, Optional


def visualize_3d_objs(objs: Sequence[Any]) -> None:
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Output viz')
    for obj in objs:
        vis.add_geometry(obj)

    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    ctr.set_front([0, 0, -1])
    ctr.set_up([0, 1, 0])
    vis.poll_events()
    vis.update_renderer()
    vis.run()
    vis.destroy_window()


def create_grasp_mesh(
    center_point: np.ndarray,
    width: float = 0.05,
    height: float = 0.1,
    depth: float = 0.05,
    gripper_distance: float = 0.105,
    gripper_height: float = 0.1,
    color_left: list = [1, 0, 0],  # Red
    color_right: list = [0, 1, 0],  # Green,
    scale: float = 1.0,
    rotation_matrix: Optional[np.ndarray] = None
) -> Sequence[o3d.geometry.TriangleMesh]:
    """
    Creates a mesh representation of a robotic gripper.
    The gripper is created aligned with PyBullet's world coordinate system:
    - Default orientation (no rotation) has gripper pointing up (+Z)
    - Fingers are aligned along Y-axis
    - Gripper width is along X-axis

    Args:
        center_point: Central position of the gripper in 3D space
        width: Width of each gripper finger
        height: Height of each gripper finger
        depth: Depth of each gripper finger
        gripper_distance: Distance between gripper fingers
        gripper_height: Height of the gripper base
        color_left: RGB color values for left finger [0-1]
        color_right: RGB color values for right finger [0-1]
        scale: Scaling factor for the gripper dimensions
        rotation_matrix: Optional 3x3 rotation matrix for gripper orientation

    Returns:
        list: List of mesh geometries representing the gripper components
    """
    grasp_geometries = []

    # Apply scaling to dimensions
    width *= scale
    height *= scale
    depth *= scale
    gripper_distance *= scale
    gripper_height *= scale

    # Create left finger
    left_finger = o3d.geometry.TriangleMesh.create_box(
        width=width/2,
        height=depth,  # Swap height and depth
        depth=height   # to align with PyBullet
    )
    left_finger.paint_uniform_color(color_left)
    # Translate along x and z axes instead of x and y
    left_finger.translate((-gripper_distance-width/2, 0, 0) + center_point)
    if rotation_matrix is not None:
        left_finger.rotate(rotation_matrix, center=center_point)
    grasp_geometries.append(left_finger)

    # Create right finger
    right_finger = o3d.geometry.TriangleMesh.create_box(
        width=width/2,
        height=depth,  # Swap height and depth
        depth=height   # to align with PyBullet
    )
    right_finger.paint_uniform_color(color_right)
    # Translate along x and z axes
    right_finger.translate((gripper_distance, 0, 0) + center_point)
    if rotation_matrix is not None:
        right_finger.rotate(rotation_matrix, center=center_point)
    grasp_geometries.append(right_finger)

    # Create coupler (horizontal bar)
    coupler = o3d.geometry.TriangleMesh.create_box(
        width=2*gripper_distance + width,
        height=depth,      # Swap height and depth
        depth=width/2     # to align with PyBullet
    )
    coupler.paint_uniform_color([0, 0, 1])
    # Translate along x and z axes
    coupler.translate((-gripper_distance-width/2, 0, gripper_height) + center_point)
    if rotation_matrix is not None:
        coupler.rotate(rotation_matrix, center=center_point)
    grasp_geometries.append(coupler)

    # Create vertical stick
    stick = o3d.geometry.TriangleMesh.create_box(
        width=width/2,
        height=depth,        # Swap height and depth
        depth=height*1.5    # to align with PyBullet
    )
    stick.paint_uniform_color([0, 0, 1])
    # Translate along x and z axes
    stick.translate((-width/4, 0, gripper_height) + center_point)
    if rotation_matrix is not None:
        stick.rotate(rotation_matrix, center=center_point)
    grasp_geometries.append(stick)

    return grasp_geometries
