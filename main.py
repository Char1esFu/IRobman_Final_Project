import os
import glob
import yaml
import time
import numpy as np
import argparse
import pybullet as p
import open3d as o3d

from typing import Dict, Any

from pybullet_object_models import ycb_objects  # type:ignore

from src.simulation import Simulation
from src.point_cloud.point_cloud import PointCloudCollector
from src.point_cloud.bounding_box import BoundingBox
from src.path_planning.planning_executor import PlanningExecutor
from src.grasping.grasping import GraspExecution

def run_exp(config: Dict[str, Any]):
    # Example Experiment Runner File
    print("Simulation Start:")
    print(config['world_settings'], config['robot_settings'])
    object_root_path = ycb_objects.getDataPath()
    files = glob.glob(os.path.join(object_root_path, "Ycb*"))
    obj_names = [file.split('/')[-1] for file in files]
    sim = Simulation(config)
    for obj_name in obj_names:
        for tstep in range(10):
            sim.reset(obj_name)
            print((f"Object: {obj_name}, Timestep: {tstep},"
                   f" pose: {sim.get_ground_tuth_position_object}"))
            pos, ori = sim.robot.pos, sim.robot.ori
            print(f"Robot inital pos: {pos} orientation: {ori}")
            l_lim, u_lim = sim.robot.lower_limits, sim.robot.upper_limits
            print(f"Robot Joint Range {l_lim} -> {u_lim}")
            sim.robot.print_joint_infos()
            jpos = sim.robot.get_joint_positions()
            print(f"Robot current Joint Positions: {jpos}")
            jvel = sim.robot.get_joint_velocites()
            print(f"Robot current Joint Velocites: {jvel}")
            ee_pos, ee_ori = sim.robot.get_ee_pose()
            print(f"Robot End Effector Position: {ee_pos}")
            print(f"Robot End Effector Orientation: {ee_ori}")
            for i in range(10000):
                sim.step()
                # for getting renders
                # rgb, depth, seg = sim.get_ee_renders()
                # rgb, depth, seg = sim.get_static_renders()
                obs_position_guess = np.zeros((2, 3))
                print((f"[{i}] Obstacle Position-Diff: "
                       f"{sim.check_obstacle_position(obs_position_guess)}"))
                goal_guess = np.zeros((7,))
                print((f"[{i}] Goal Obj Pos-Diff: "
                       f"{sim.check_goal_obj_pos(goal_guess)}"))
                print(f"[{i}] Goal Satisfied: {sim.check_goal()}")
    sim.close()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Robot point cloud collection and bounding box calculation')
    # All objects: 
    # Low objects: YcbBanana, YcbFoamBrick, YcbHammer, YcbMediumClamp, YcbPear, YcbScissors, YcbStrawberry, YcbTennisBall, 
    # Medium objects: YcbGelatinBox, YcbMasterChefCan, YcbPottedMeatCan, YcbTomatoSoupCan
    # High objects: YcbCrackerBox, YcbMustardBottle, 
    # Unstable objects: YcbChipsCan, YcbPowerDrill
    parser.add_argument('--object', type=str, default="YcbGelatinBox",
                        help='Target object name')
    parser.add_argument('--no-vis', action='store_true',
                        help='Disable point cloud visualization')
    parser.add_argument('--no-grasp', action='store_true',
                        help='Disable grasp execution')
    parser.add_argument('--no-planning', action='store_true',
                        help='Disable path planning')
    parser.add_argument('--planning-type', type=str, choices=['joint', 'cartesian'], default='joint',
                        help='Select path planning type: joint (joint space) or cartesian (Cartesian space)')
    
    args = parser.parse_args()
    
    # Load configuration file
    config_path = "configs/test_config.yaml"
    with open(config_path, "r") as stream:
        try:
            config = yaml.safe_load(stream)
            print(f"Loading configuration file: {config_path}")
        except yaml.YAMLError as exc:
            print(f"Configuration file loading error: {exc}")
    
    # Variable initialization
    sim = None
    collector = None
    point_clouds = None
    bbox = None
    grasp_success = False
    grasp_executor = None
    
    # Step 1: Collect point cloud
    print("Step 1: Starting point cloud collection...")
    sim = Simulation(config)
    collector = PointCloudCollector(config, sim)
    point_clouds = collector.collect_point_clouds(args.object)
    print(f"Successfully collected {len(point_clouds)} point clouds.")
    
    # Check and print the maximum z-axis point from the high viewpoint cloud
    for data in point_clouds:
        if data.get('viewpoint_idx') == 'high_point' and 'max_z_point' in data:
            print(f"\nMaximum z-axis point from the high viewpoint cloud: {data['max_z_point']}")
    
    # Step 2: Compute and visualize bounding box
    if point_clouds:
        bbox = BoundingBox.compute_point_cloud_bbox(sim, collector, point_clouds, not args.no_vis)
        
        # Step 3: Execute grasp (unless --no-grasp flag is set)
        if not args.no_grasp and bbox is not None:
            grasp_executor = GraspExecution(sim)
            grasp_success, grasp_executor = grasp_executor.execute_complete_grasp(bbox, point_clouds, True)
            
            # Step 4: Execute path planning (if grasp succeeded and planning not disabled)
            if grasp_success and not args.no_planning and grasp_executor is not None:
                # Create path planning executor
                planning_executor = PlanningExecutor(sim, config)
                planning_success = planning_executor.execute_planning(
                    grasp_executor, 
                    planning_type=args.planning_type,
                    visualize=True
                )
                if planning_success:
                    print(f"\n{'Cartesian space' if args.planning_type == 'cartesian' else 'Joint space'} path planning executed successfully!")
                else:
                    print(f"\n{'Cartesian space' if args.planning_type == 'cartesian' else 'Joint space'} path planning execution failed.")
    
    input("\nPress Enter to close the simulation...")
    
    # Close simulation
    if sim is not None:
        sim.close()
