from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs import mdp
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import mdp

from isaaclab.utils.math import axis_angle_from_quat

import isaacsim.core.utils.torch as torch_utils
from envs.factory.manager.mdp.rewards import currently_inrange
from envs.factory.manager.mdp.events import compute_keypoint_value, get_handheld_asset_relative_pose
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def global_timeout(
    env: ManagerBasedRLEnv
):
    
    if env.common_step_counter % env.max_episode_length == 0:
        return torch.ones((env.num_envs,), device=env.device, dtype=torch.bool)
    return torch.zeros((env.num_envs,), device=env.device, dtype=torch.bool)

def factory_success(
    env: ManagerBasedRLEnv
):
    # check if in success postion
    in_range = currently_inrange(
        env, 
        success_threshold = env.cfg_task.success_threshold
    )


    # check if arm is not moving
    is_still = torch.where(torch.linalg.norm(mdp.joint_vel_rel(env),  axis=1) < 1.0e-3, True, False)
    if torch.logical_and(in_range, is_still).any():
        print("Success:", torch.logical_and(in_range, is_still))
        import time
        time.sleep(10)

    return torch.logical_and(in_range, is_still)

def force_check(
    env: ManagerBasedRLEnv,
    threshold: float = 1.0e6
):
    fts = env.scene['force_torque_sensor']
    if fts.history_length > 1:
        raw = env.scene['force_torque_sensor'].data.net_forces_w_history
        mag = torch.linalg.norm(raw,dim=1)
        val = torch.max(raw, dim=1)
        return torch.where( val > threshold, True, False)
    else:
        mag = torch.linalg.norm(env.scene['force_torque_sensor'].data.net_forces_w, dim=1)
        return torch.where(mag > threshold, True, False) 
    

def in_ws_limits(
    env: ManagerBasedRLEnv,
    pose_range: dict
):
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    mins = torch.tensor([range_list[i][0] for i in range(len(range_list)) ])
    maxs = torch.tensor([range_list[i][1] for i in range(len(range_list)) ])

    robot: Articulation = env.scene["robot"]
    robot_pos = (
        robot.data.body_link_pos_w[:, env.fingertip_body_idx] - env.scene.env_origins
    )
    robot_quat = robot.data.body_link_quat_w[:, env.fingertip_body_idx]
    robot_ea = torch_utils.euler_xyz_from_quat(robot_quat)
    in_pos = torch.logical_and( robot_pos < maxs[:3], robot_pos > mins[:3])
    in_rot = torch.logical_and( robot_ea < maxs[3:],  robot_ea > mins[3:])

    return torch.logical_or(~in_pos, ~in_rot)

def dropped_held_asset(
    env: ManagerBasedRLEnv,
    max_dist: float = 0.05, # 5 cm
    max_rot: float = 3.14159 / 6 # pi/6=30deg
):
    # get current locations

    compute_keypoint_value(env) # defines held_pos and held_quat

    robot: Articulation = env.scene["robot"]
    robot_pos = (
        robot.data.body_link_pos_w[:, env.fingertip_body_idx] - env.scene.env_origins
    )
    robot_quat = robot.data.body_link_quat_w[:, env.fingertip_body_idx]   

    # get ideal transform
    ideal_pos, ideal_quat = get_handheld_asset_relative_pose(env)

    # current difference minus the ideal distance
    dpos = torch.linalg.norm(robot_pos - env.held_pos - ideal_pos)

    rot_diff_quat = torch_utils.quat_mul(
        robot_quat, torch_utils.quat_conjugate(env.held_quat)
    )
    rot_diff_quat = torch_utils.quat_mul(
        rot_diff_quat, torch_utils.quat_conjugate(ideal_quat)
    )
    drot = torch.norm(axis_angle_from_quat(rot_diff_quat), dim=-1)

    return torch.logical_or(dpos, drot)

    

