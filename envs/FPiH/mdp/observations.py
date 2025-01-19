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

def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("peg"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )
    return object_pos_b

def frame_in_robot_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    frame_name: str = "peg_end_frame"
):
    robot: RigidObject = env.scene[robot_cfg.name]
    frame_pose = env.scene[frame_name].data.target_pos_w[:, 0, :]
    frame_quat = env.scene[frame_name].data.target_quat_w[:,0,:]
    if frame_name == "peg_end_frame":
        # add the peg offset
        if type(env.cfg.peg_offsets) == list:
            peg_offsets = torch.zeros((env.num_envs, 3), device="cuda:0")
    
            for i in range(env.num_envs):
                peg_offsets[i, 2] = env.cfg.peg_offsets[i % len(env.cfg.peg_offsets)] / 2.0

            env.cfg.peg_offsets = peg_offsets
            
            env.cfg.peg_offsets.to(env.device)
        frame_pose += quat_apply(frame_quat, env.cfg.peg_offsets)
    object_pos_b, object_pos_w = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], frame_pose, frame_quat
    )
    return torch.cat( (object_pos_b, object_pos_w), dim=1 )


def force_torque_sensor(
        env: ManagerBasedRLEnv,
        frame: str = "panda_joint_1"
):
    if env.scene.cfg.robot_av is None:
        return torch.zeros((env.num_envs, 6), dtype=torch.float32, device=env.device)

    return env.scene.cfg.robot_av.get_measured_joint_forces()[:,8,:]


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



def is_peg_inserted(
    env: ManagerBasedRLEnv,
    xy_thresh: float = 0.01,
    z_thresh: float = 0.005
):
    hole_pos = env.scene['hole_frame'].data.target_pos_w[:,0,:]
    
    peg_pos = env.scene["peg_end_frame"].data.target_pos_w[:, 0, :]
    peg_quat = env.scene["peg_end_frame"].data.target_quat_w[:,0,:]

    peg_pos += quat_apply(peg_quat, env.cfg.peg_offsets)

    dxy = torch.linalg.norm( peg_pos[:,:2] - hole_pos[:,:2], dim=1)
    
    dz = torch.abs(peg_pos[:,2] - hole_pos[:,2])

    
    return torch.logical_and(dxy < xy_thresh, dz < 0.05)


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