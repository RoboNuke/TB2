from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.envs import mdp

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

def update_z_low(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor
):
    print("called update_z_low")
    if 'log' in env.extras.keys() and "Episode_Termination/success" in env.extras['log']:
        last_success_rate = env.extras['log']["Episode_Termination/success"] / env.num_envs
        if last_success_rate > 0.8:
            # increase min z
            env.z_low = min(env.cfg_task.z_increase+env.z_low, env.cfg_task.hand_init_pos[2])
            return {"Curriculum/z_low" : env.z_low}
        elif last_success_rate < 0.1:
            # decrease min z
            env.z_low = max( 
                env.z_low - env.cfg_task.z_decrease, 
                env.cfg_task.success_threshold * env.cfg_task.fixed_asset_cfg.height+0.001 # add 1 mm so we can't start at success
            )
            return {"Curriculum/z_low" : env.z_low}

