import numpy as np
import open3d as o3d
from typing import Any, Sequence


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