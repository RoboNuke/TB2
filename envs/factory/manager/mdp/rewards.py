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
        violated = torch.where( val > threshold, 1.0, 0.0)
    else:
        val = torch.linalg.norm(env.scene['force_torque_sensor'].data.net_forces_w, dim=1)
        violated = torch.where(mag > threshold, 1.0, 0.0) 

    if 'my_log_data' not in env.extras:
        env.extras['my_log_data'] = {}
    env.extras['my_log_data']['force_mag'] = val
    env.extras['my_log_data']['force_violations'] = violated

    return violated

def squashing_fn(x, a, b):
    return 1 / (torch.exp(a * x) + b + torch.exp(-a * x))


def keypoint_reward(
    env: ManagerBasedRLEnv,
    a: float = 100.0,
    b: float = 0.0,
    rew_name: str = "keypoint_reward"
):
    compute_keypoint_value(env)#, dt=env.physics_dt))
    rew = squashing_fn(env.keypoint_dist, a, b)
    if 'my_log_data' not in env.extras.keys():
        env.extras['my_log_data'] = {}
    if 'step_rew' not in env.extras['my_log_data']:
        env.extras['my_log_data']['step_rew'] = {} 

    env.extras['my_log_data']['step_rew'][rew_name] = rew
    
    return rew

def currently_inrange(
    env: ManagerBasedRLEnv,
    success_threshold: float = 0.01,
    check_rot: bool = False,
    rew_name: str = "keypoint_reward"
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
        env.extras['my_log_data'] = {}
    if 'count_step'  not in env.extras['my_log_data'].keys():
        env.extras['my_log_data']['count_step'] = {}
        env.extras['my_log_data']['once'] = {}
    if rew_name not in env.extras['my_log_data']['count_step']:
        env.extras['my_log_data']['count_step'][rew_name] = torch.zeros_like(curr_successes)
        env.extras['my_log_data']['once'][rew_name] = torch.zeros_like(curr_successes)
    env.extras['my_log_data']['count_step'][rew_name] = curr_successes
    env.extras['my_log_data']['once'][rew_name] = torch.logical_or(env.extras['my_log_data']['once'][rew_name], curr_successes)
    

    return curr_successes
     