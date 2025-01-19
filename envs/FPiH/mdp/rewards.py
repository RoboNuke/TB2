from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import subtract_frame_transforms
from omni.isaac.core.articulations import ArticulationView


from omni.isaac.lab.utils.math import quat_apply

def peg_hole_xy(
    env: ManagerBasedRLEnv
):
    goal_pos = env.scene["hole_frame"].data.target_pos_w[:, 0, :]
    
    peg_pos = env.scene["peg_end_frame"].data.target_pos_w[:, 0, :]
    peg_quat = env.scene["peg_end_frame"].data.target_quat_w[:,0,:]

    peg_pos += quat_apply(peg_quat, env.cfg.peg_offsets)

    dist = torch.linalg.norm(goal_pos - peg_pos, axis=1)
    #print(goal_pos[0, :], peg_pos[0,:], dist[0], 1 - torch.tanh(5 * dist[0]))
    return (1 - torch.tanh(5 * dist))


def maniskill_reward(
    env: ManagerBasedRLEnv
):
    
    # reach reward (if robot drops peg still gets a reward signal)
    goal_pos = env.scene["hole_frame"].data.target_pos_w[:, 0, :]
    ee_pos = env.scene['ee_frame'].data.target_pos_w[:, 0, :]
    
    peg_head_pos = env.scene["peg_end_frame"].data.target_pos_w[:, 0, :]
    peg_head_quat = env.scene["peg_end_frame"].data.target_quat_w[:,0,:]

    peg_head_pos += quat_apply(peg_head_quat, env.cfg.peg_offsets)

    peg_pos = env.scene['peg'].data.root_pos_w

    dist_tcp = torch.linalg.norm(
        goal_pos[:,:2] - ee_pos[:,:2],
        axis = 1
    )

    reward = (1 - torch.tanh(5 * dist_tcp)) 

    # this reward encourage holding onto the peg and 
    # keeping the peg aligned vertically
    dist_xy = torch.linalg.norm(
        goal_pos[:,:2] - peg_head_pos[:,:2], 
        axis=1
    )

    dist2_xy = torch.linalg.norm(
        goal_pos[:,:2] - peg_pos[:,:2],
        axis=1
    )

    pre_insert_reward = 4 * (
        1 - torch.tanh(
            0.5 * (dist2_xy + dist_xy) + 
            4.5 * torch.maximum(dist2_xy, dist_xy)
        )
    )

    reward += pre_insert_reward 

    # finally reward it for pushing it down
    pre_inserted = torch.logical_and(dist_xy < 0.01, dist2_xy < 0.01)
    # make sure that the peg is pointed down
    #print("pre:", pre_inserted)
    pre_inserted = torch.logical_and(
        pre_inserted, 
        peg_pos[:,2] > peg_head_pos[:,2]
    )
    #print(pre_inserted)
    #dist = torch.linalg.norm(
    #    goal_pos - peg_head_pos, 
    #    axis=1
    #)
    dist = peg_head_pos[:,2] 
    #print(f"r:{5 * (1 - torch.tanh(5*dist))}\tz-dist:{dist}\tpi:{pre_inserted}")
    reward += 5 * (1 - torch.tanh(5*dist)) * pre_inserted

    #reward[info["success"]] = 10.0
    #print(f"g:{is_grasped}\to:{is_over_box}\td:{dist}\tr:{reward}")
    return reward / 10.0
