from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

from envs.factory.manager.mdp.events import compute_keypoint_value


def force_check(
    env: ManagerBasedRLEnv,
    threshold: float = 1.0e6
):
    fts = env.scene['force_torque_sensor']
    if fts.history_length > 1:
        raw = env.scene['force_torque_sensor'].data.net_forces_w_history
        mag = torch.linalg.norm(raw,dim=1)
        val = torch.max(raw, dim=1)
        env.extras['my_log_data']['force_mag'] = val
        return torch.where( val > threshold, 1.0, 0.0)
    else:
        mag = torch.linalg.norm(env.scene['force_torque_sensor'].data.net_forces_w, dim=1)
        env.extras['my_log_data']['force_mag'] = mag
        return torch.where(mag > threshold, 1.0, 0.0) 

def squashing_fn(x, a, b):
    return 1 / (torch.exp(a * x) + b + torch.exp(-a * x))


def keypoint_reward(
    env: ManagerBasedRLEnv,
    a: float = 100.0,
    b: float = 0.0
):
    compute_keypoint_value(env)#, dt=env.physics_dt))
    return squashing_fn(env.keypoint_dist, a, b)

def currently_inrange(
    env: ManagerBasedRLEnv,
    success_threshold: float = 0.01,
    check_rot: bool = False
):
    compute_keypoint_value(env)#, dt=env.physics_dt))
    curr_successes = torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device)

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
    
    if 'my_log_data' not in env.extras.keys():
        env.extras['my_log_data'] = {
            'success_once':torch.zeros_like(curr_successes),
            'engaged_once':torch.zeros_like(curr_successes),
            'count_step':{
                'success':torch.zeros_like(curr_successes),
                'engaged':torch.zeros_like(curr_successes)
            },
            'once':{
                'success':torch.zeros_like(curr_successes),
                'engaged':torch.zeros_like(curr_successes)
            }
        }

    name = 'success' if success_threshold < 0.5 else 'engaged'
    env.extras['my_log_data']['count_step'][name] = curr_successes
    env.extras['my_log_data']['once'][name] = torch.logical_or(env.extras['my_log_data']['once'][name], curr_successes)
    

    return curr_successes
     