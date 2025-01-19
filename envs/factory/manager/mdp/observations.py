from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import subtract_frame_transforms, quat_apply
from omni.isaac.core.articulations import ArticulationView


from omni.isaac.lab.envs import mdp
import cv2

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

from envs.factory.manager.mdp.events import compute_intermediate_values

def scaled_jnt_pos_rel(
        env: ManagerBasedRLEnv
):
    robot: Articulation = env.scene['robot']
    lims = robot.data.joint_limits
    bot = lims[:,:,0]
    top = lims[:,:,1]
    joint_pos_rel = mdp.joint_pos_rel(env)
    scaled = (joint_pos_rel - bot) / (top - bot )
    return scaled


def held_asset_pose(
        env: ManagerBasedRLEnv
):
    try:
        compute_intermediate_values(env, dt=env.physics_dt)
        return torch.cat([env.held_pos, env.held_quat], dim=1)
    except:
        return torch.zeros(env.num_envs, 7)

def fixed_asset_pose(
        env: ManagerBasedRLEnv
):
    try:
        compute_intermediate_values(env, dt=env.physics_dt)
        return torch.cat([env.fixed_pos, env.fixed_quat], dim=1)
    except:
        return torch.zeros(env.num_envs, 7)

def force_torque_sensor(
        env: ManagerBasedRLEnv,
        frame: str = "panda_joint_1"
):
    
    try:
        return env.robot_av.get_measured_joint_forces()[:,8,:]
    except:
        return torch.zeros((env.num_envs, 6), dtype=torch.float32, device=env.device)

def peg_contact_sensor(
    env: ManagerBasedRLEnv
):
    return torch.linalg.norm(env.scene['peg_contact_force'].data.net_forces_w[:,0,:], dim=1)

def camera_image(
    env: ManagerBasedRLEnv
):
    if env.cfg.recording:
        cam_data = env.scene['tiled_camera'].data
        return cam_data.output['rgb']
    else:
        try:
            return env.cfg.empty_img
        except AttributeError:
            env.cfg.empty_img = torch.zeros(
                (env.num_envs, 180, 240, 3),
                device = env.device
            )
            return env.cfg.empty_img
        
def joint_acc_rel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """The joint acceleration of the asset w.r.t. the default joint accelerations.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their accelerations returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_acc[:, asset_cfg.joint_ids]


def scaled_jnt_pos_rel(
        env: ManagerBasedRLEnv
):
    robot: Articulation = env.scene['robot']
    lims = robot.data.joint_limits
    bot = lims[:,:,0]
    top = lims[:,:,1]
    joint_pos_rel = mdp.joint_pos_rel(env)
    scaled = (joint_pos_rel - bot) / (top - bot )
    return scaled

def scaled_jnt_vel_rel(
        env: ManagerBasedRLEnv
):
    robot: Articulation = env.scene['robot']
    lims = robot.data.joint_velocity_limits

    joint_vel_rel = mdp.joint_vel_rel(env)

    scaled = joint_vel_rel / lims
    #print("vel:", joint_vel_rel, lims, scaled)
    return scaled