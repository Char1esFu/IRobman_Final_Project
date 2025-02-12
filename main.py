import os
import glob
import yaml
import time

import numpy as np
import pybullet as p
from typing import Dict, Any

from pybullet_object_models import ycb_objects  # type:ignore
from src.ik_solver import DifferentialIKSolver
from src.simulation import Simulation
from src.rrt_planner import RRTStarConnect
from src.obstacle_tracker import ObstacleTracker

def run_exp(config: Dict[str, Any]):
    print("Simulation Start:")
    print(config['world_settings'], config['robot_settings'])
    object_root_path = ycb_objects.getDataPath()
    files = glob.glob(os.path.join(object_root_path, "Ycb*"))
    obj_names = [file.split('/')[-1] for file in files]
    sim = Simulation(config)
    
    # track init
    tracker = None
    debug_ids = []
    if config['world_settings']['turn_on_obstacles']:
        tracker = ObstacleTracker(n_obstacles=2, exp_settings=config)
    
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
            
            # 1. 获取起始配置(当前关节状态)
            start_config = jpos.copy()  # 使用copy避免引用

            # 2. 获取目标位置
            min_lim, max_lim = sim.goal._get_goal_lims()
            goal_pos = np.array([
                (min_lim[0] + max_lim[0])/2,
                (min_lim[1] + max_lim[1])/2,
                max_lim[2] + 0.2
            ])

            goal_pos[0] -= 0.2
            goal_pos[1] -= 0.2
            
            # 3. 可视化目标位置
            if hasattr(sim, 'goal_visual_id'):
                p.removeUserDebugItem(sim.goal_visual_id)
            sim.goal_visual_id = p.addUserDebugPoints(
                [goal_pos],
                [[0, 1, 0]],  # 绿色
                pointSize=6
            )

            print(f"\nPlanning path from {ee_pos} to {goal_pos}")

            # 4. 临时储存当前关节状态
            original_joint_positions = sim.robot.get_joint_positions()

            # 5. 求解目标IK
            ik_solver = DifferentialIKSolver(
                robot_id=sim.robot.id,
                ee_link_index=sim.robot.ee_idx,
                damping=0.05  # 增大阻尼系数
            )
            goal_config = ik_solver.solve(
                goal_pos,
                p.getQuaternionFromEuler([0, np.pi/2, np.pi/2]),
                original_joint_positions,
                max_iters=50,  # 增加最大迭代次数
                tolerance=0.01  # 降低误差容忍度
            )

            # 6. 恢复初始关节状态
            for idx, pos in zip(sim.robot.arm_idx, original_joint_positions):
                p.resetJointState(sim.robot.id, idx, pos)

            # 7. 初始化规划器并规划
            planner = RRTStarConnect(
                robot=sim.robot,
                obstacle_tracker=tracker,
                goal_region={'position': goal_pos},
                step_size=0.1, 
                neighbor_radius=0.5, 
                goal_bias=0 
            )

            path = planner.plan(start_config, goal_config)
            
            # 8. 执行路径或原有循环
            path_index = 0
            for i in range(10000):
                # 获取障碍物预测位置
                if tracker:
                    # 清除旧可视化
                    if debug_ids:
                        for debug_id in debug_ids:
                            p.removeUserDebugItem(debug_id)
                    
                    # 更新追踪器
                    rgb, depth, seg = sim.get_static_renders()
                    detected = tracker.detect_obstacles(rgb, depth, seg)
                    
                    pred_pos = tracker.update(detected)
                    debug_ids = tracker.visualize_tracking_3d(pred_pos)
                else:
                    pred_pos = np.zeros((2, 3))
                
                # 使用追踪数据
                print(f"[{i}] Obstacle Position-Diff: {sim.check_obstacle_position(pred_pos)}")
                
                if path is not None and path_index < len(path):
                    sim.robot.position_control(path[path_index])
                    path_index += 1

                sim.step()
                ee_pos, ee_ori = sim.robot.get_ee_pose()
                print(f"[{i}] End Effector Position: {ee_pos}")
                print(f"[{i}] End Effector Orientation: {ee_ori}")
                
                if sim.obstacles_flag:
                    obs_position_guess = pred_pos
                else: 
                    obs_position_guess = np.zeros((2, 3))
                    
                print(f"[{i}] Obstacle Position-Diff: {sim.check_obstacle_position(obs_position_guess)}")
                goal_guess = np.zeros((7,))
                print((f"[{i}] Goal Obj Pos-Diff: "
                       f"{sim.check_goal_obj_pos(goal_guess)}"))
                print(f"[{i}] Goal Satisfied: {sim.check_goal()}")

                if sim.check_goal():
                    print("Reached goal!")
                    break
    
    sim.close()


if __name__ == "__main__":
    with open("configs/test_config.yaml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
            print(config)
        except yaml.YAMLError as exc:
            print(exc)
    run_exp(config)
