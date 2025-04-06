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
    #print("called update_z_low")
    if env.evaluating:
        step = env.num_envs // env.cfg.num_agents
        succ_rews = torch.unsqueeze(env.reward_manager._episode_sums['success'], 1).clone()
        out = {}
        for i in range(env.cfg.num_agents):
            l = i * step
            h = (i+1) * step
            succ_envs = torch.where(succ_rews[l:h] > 0.0001, 1, 0)
            sr = torch.sum(succ_envs) / step
            if sr > 0.8:
                # increase min z
                env.z_low[l:h] = env.cfg_task.z_increase + env.z_low[l:h]
                #print(f"\tIncrease z_low[{i}]:", env.z_low[l].item())
                #return {"Curriculum/z_low" : env.z_low}
            elif sr < 0.1:
                # decrease min z
                env.z_low[l:h] = env.z_low[l:h] - env.cfg_task.z_decrease
                #print(f"\tDecrease z_low[{i}]:", env.z_low[l].item())
                #return {"z_low" : env.z_low}
            #else:
            #    print(f"\tUnchanged z_low[{i}]:", env.z_low[l].item())

            out[f'agent_{i}'] = env.z_low[l]
        
        lowest_z = env.cfg_task.success_threshold * env.cfg_task.fixed_asset_cfg.height+0.001 # add 1 mm so we can't start at success
        env.z_low[env.z_low > env.cfg_task.hand_init_pos[2]] = env.cfg_task.hand_init_pos[2]
        env.z_low[env.z_low < lowest_z] = lowest_z
        #print("\tFinal:", env.z_low)
        return out


