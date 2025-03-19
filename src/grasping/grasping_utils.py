import numpy as np
import open3d as o3d
from typing import Any, Sequence, Optional


def visualize_3d_objs(objs: Sequence[Any], show_world_frame: bool = True) -> None:
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Output viz')
    
    # Add world coordinate frame if requested
    if show_world_frame:
        world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
        vis.add_geometry(world_frame)
    
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
    scale: float = 0.6,
    rotation_matrix: Optional[np.ndarray] = None
) -> Sequence[o3d.geometry.TriangleMesh]:
    """
    Creates a mesh representation of a robotic gripper.

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
        height=height,
        depth=depth
    )
    left_finger.paint_uniform_color(color_left)
    left_finger.translate((-gripper_distance-width/2, 0, 0) + center_point)
    if rotation_matrix is not None:
        left_finger.rotate(rotation_matrix, center=center_point)
    grasp_geometries.append(left_finger)

    # Create right finger
    right_finger = o3d.geometry.TriangleMesh.create_box(
        width=width/2,
        height=height,
        depth=depth
    )
    right_finger.paint_uniform_color(color_right)
    right_finger.translate((gripper_distance, 0, 0) + center_point)
    if rotation_matrix is not None:
        right_finger.rotate(rotation_matrix, center=center_point)
    grasp_geometries.append(right_finger)

    coupler = o3d.geometry.TriangleMesh.create_box(
        width=2*gripper_distance + width,
        height=width/2,
        depth=depth
    )
    coupler.paint_uniform_color([0, 0, 1])
    coupler.translate(
        (-gripper_distance-width/2, gripper_height, 0) + center_point)
    if rotation_matrix is not None:
        coupler.rotate(rotation_matrix, center=center_point)
    grasp_geometries.append(coupler)

    stick = o3d.geometry.TriangleMesh.create_box(
        width=width/2,
        height=height*1.5,
        depth=depth
    )
    stick.paint_uniform_color([0, 0, 1])
    stick.translate((-width/4, gripper_height, 0) + center_point)
    if rotation_matrix is not None:
        stick.rotate(rotation_matrix, center=center_point)
    grasp_geometries.append(stick)

    return grasp_geometries