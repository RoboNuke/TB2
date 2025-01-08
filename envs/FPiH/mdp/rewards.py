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

    dist = torch.linalg.norm(goal_pos - peg_pos, dim=1)
    #print(goal_pos[0, :], peg_pos[0,:], dist[0], 1 - torch.tanh(5 * dist[0]))
    return (1 - torch.tanh(5 * dist))

