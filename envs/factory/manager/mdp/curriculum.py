from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs import mdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def update_z_low(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor
):
    #print("called update_z_low")
    if True: #env.evaluating:
        out = {}
        step = env.num_envs // env.cfg.num_agents
        if 'my_log_data' in env.extras:
            success_once = env.extras['my_log_data']['success_once']
            for i in range(env.cfg.num_agents):
                l = i * step
                h = (i+1) * step
                sr = torch.sum(success_once[l:h]) / step
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
        else:
            print("skipped curriculum")
        lowest_z = env.cfg_task.success_threshold * env.cfg_task.fixed_asset_cfg.height+0.001 # add 1 mm so we can't start at success
        env.z_low[env.z_low > env.cfg_task.hand_init_pos[2]] = env.cfg_task.hand_init_pos[2]
        env.z_low[env.z_low < lowest_z] = lowest_z
        for i in range(env.cfg.num_agents):
            l = i * step
            out[f'z_low_agent_{i}'] = env.z_low[l]
        #print(torch.min(env.z_low))
        return out


