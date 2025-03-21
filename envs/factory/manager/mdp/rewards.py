from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg

from envs.factory.manager.mdp.events import compute_keypoint_value

def squashing_fn(x, a, b):
    return 1 / (torch.exp(a * x) + b + torch.exp(-a * x))


def keypoint_reward(
    env: ManagerBasedRLEnv,
    a: float = 100.0,
    b: float = 0.0
):
    compute_keypoint_value(env)#, dt=env.physics_dt))
    
    #check if below the hole and not centered on it, give zero reward if that is the case
    """
    z_disp = env.held_base_pos[:, 2] - env.target_held_base_pos[:, 2]
    fixed_cfg = env.cfg_task.fixed_asset_cfg
    if env.cfg_task.name == "peg_insert" or env.cfg_task.name == "gear_mesh":
        height_threshold = fixed_cfg.height * 0.9
    elif env.cfg_task.name == "nut_thread":
        height_threshold = fixed_cfg.thread_pitch * 0.9
    else:
        raise NotImplementedError("Task not implemented")
    
    # check outside hole range
    xy_dist = torch.linalg.vector_norm(env.target_held_base_pos[:, 0:2] - env.held_base_pos[:, 0:2], dim=1)
    not_centered = torch.where(xy_dist < 0.0025, torch.zeros_like(raw_keypoint_reward), torch.ones_like(raw_keypoint_reward))

    #keypoint_adjusted = torch.where(
    #    torch.logical_and(z_disp < height_threshold, not_centered), 
    #    raw_keypoint_reward, 
    #    torch.zeros_like(raw_keypoint_reward)
    #)
    raw_keypoint_reward[~torch.logical_and( z_disp < height_threshold, not_centered)] = 0.0
    """

    #raw_keypoint_reward = squashing_fn(env.adj_keypoint_dist, a, b) # 
    return squashing_fn(env.keypoint_dist, a, b)

def currently_inrange(
    env: ManagerBasedRLEnv,
    success_threshold: float = 0.01,
    check_rot: bool = False
):
    compute_keypoint_value(env)#, dt=env.physics_dt))
    curr_successes = torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device)
    """
    xy_dist = torch.linalg.vector_norm(env.target_held_base_pos[:, 0:2] - env.held_base_pos[:, 0:2], dim=1)
    z_disp = env.held_base_pos[:, 2] - env.target_held_base_pos[:, 2]

    is_centered = torch.where(xy_dist < 0.0025, torch.ones_like(curr_successes), torch.zeros_like(curr_successes))
    """
    # Height threshold to target
    fixed_cfg = env.cfg_task.fixed_asset_cfg
    if env.cfg_task.name == "peg_insert" or env.cfg_task.name == "gear_mesh":
        height_threshold = fixed_cfg.height * success_threshold
    elif env.cfg_task.name == "nut_thread":
        height_threshold = fixed_cfg.thread_pitch * success_threshold
    else:
        raise NotImplementedError("Task not implemented")
    
    is_close_or_below = torch.where(
        env.z_disp < height_threshold, torch.ones_like(curr_successes), torch.zeros_like(curr_successes)
    )
    curr_successes = torch.logical_and(env.is_centered, is_close_or_below)

    if check_rot:
        is_rotated = env.curr_yaw < env.cfg_task.ee_success_yaw
        curr_successes = torch.logical_and(curr_successes, is_rotated)

    return curr_successes
     