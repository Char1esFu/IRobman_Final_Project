import numpy as np
import open3d as o3d
from typing import Any,Sequence, Optional



def create_grasp_mesh(
    center_point: np.ndarray,
    color_left: list = [1, 0, 0],  # Red
    color_right: list = [0, 1, 0],  # Green,
    rotation_matrix: Optional[np.ndarray] = None
) -> Sequence[o3d.geometry.TriangleMesh]:
    """
    Creates a mesh representation of a robotic gripper.
    
    Coordinate system aligned with PyBullet:
    - Y-axis: Gripper opening direction (left: -y, right: +y)
    - X-axis: Finger thickness direction
    - Z-axis: Finger height direction (positive z points in gripper forward direction)

    Args:
        center_point: Central position of the gripper in 3D space
        width: Width of each gripper finger (thickness in X direction)
        height: Height of each gripper finger (Z direction)
        depth: Depth of each gripper finger (Y direction - opening distance)
        gripper_distance: Distance between gripper fingers along Y axis
        gripper_height: Height of the gripper base in Z direction
        color_left: RGB color values for left finger [0-1]
        color_right: RGB color values for right finger [0-1]
        scale: Scaling factor for the gripper dimensions
        rotation_matrix: Optional 3x3 rotation matrix for gripper orientation

    Returns:
        list: List of mesh geometries representing the gripper components
    """
    grasp_geometries = []
    wid = 0.02
    hei = 0.05
    dep = 0.005

    default_rotation = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    # default_rotation = np.eye(3)

    if rotation_matrix is not None:
        final_rotation = rotation_matrix @ default_rotation
    else:
        final_rotation = default_rotation


    # Create left finger - in -Y direction
    left_finger = o3d.geometry.TriangleMesh.create_box(
        width=wid,     # X-axis - thickness
        height=dep,  # Y-axis - half depth for finger
        depth=hei     # Z-axis - height
    )
    left_finger.paint_uniform_color(color_left)
    left_finger.translate((-0.01, -0.045, 0.005) + center_point)
    left_finger.rotate(final_rotation, center=center_point)
    grasp_geometries.append(left_finger)

    # Create right finger - in +Y direction
    right_finger = o3d.geometry.TriangleMesh.create_box(
        width=wid,     # X-axis - thickness
        height=dep,  # Y-axis - half depth for finger
        depth=hei     # Z-axis - height
    )
    right_finger.paint_uniform_color(color_right)
    right_finger.translate((-0.01, 0.04, 0.005) + center_point)
    right_finger.rotate(final_rotation, center=center_point)
    grasp_geometries.append(right_finger)

    # Create coupler - horizontal bar connecting the fingers at the top
    coupler = o3d.geometry.TriangleMesh.create_box(
        width=wid,     # X-axis - thickness
        height=0.09, # Y-axis - spans across both fingers
        depth=0.01   # Z-axis - height of connector
    )
    coupler.paint_uniform_color([0, 0, 1])
    coupler.translate((-0.01, -0.045, -0.005) + center_point)
    coupler.rotate(final_rotation, center=center_point)
    grasp_geometries.append(coupler)

    # Create vertical stick
    stick = o3d.geometry.TriangleMesh.create_box(
        width=wid,   # X-axis - thickness
        height=0.01,  # Y-axis - depth
        depth=0.02 # Z-axis - height
    )
    stick.paint_uniform_color([0, 0, 1])
    stick.translate((-0.01, -0.005, -0.02) + center_point)
    stick.rotate(final_rotation, center=center_point)
    grasp_geometries.append(stick)

    return grasp_geometries

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

def visualize_gripper(
    center_point: np.ndarray = np.array([0, 0, 0]),
    scale: float = 1,
    rotation_matrix: Optional[np.ndarray] = None,
    coordinate_frame_size: float = 0.1
) -> None:
    """
    Visualizes the gripper mesh and displays a world coordinate frame.
    
    Args:
        center_point: Central position of the gripper in 3D space
        width: Width of each gripper finger
        height: Height of each gripper finger
        depth: Depth of each gripper finger
        gripper_distance: Distance between gripper fingers
        gripper_height: Height of the gripper base
        scale: Scaling factor for the gripper dimensions
        rotation_matrix: Optional 3x3 rotation matrix for gripper orientation
        coordinate_frame_size: Size of the coordinate frame
    """
    # Create gripper mesh
    gripper_meshes = create_grasp_mesh(
        center_point=center_point,
        rotation_matrix=rotation_matrix
    )
    
    # Create coordinate frame
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=coordinate_frame_size, origin=[0, 0, 0])
    
    # Create visualizer and add all geometries
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # Add coordinate frame
    vis.add_geometry(coordinate_frame)
    
    # Add all gripper components
    for mesh in gripper_meshes:
        vis.add_geometry(mesh)
    
    # Set some view options
    opt = vis.get_render_option()
    opt.background_color = np.array([0.8, 0.8, 0.8])  # Light gray background
    opt.point_size = 5.0
    
    # Run the visualizer
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    # Example usage to visualize the default gripper
    visualize_gripper()