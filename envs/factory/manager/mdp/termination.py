from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs import mdp
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import mdp

from envs.factory.manager.mdp.rewards import currently_inrange

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
    
