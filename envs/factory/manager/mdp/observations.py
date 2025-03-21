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

from envs.factory.manager.mdp.events import compute_keypoint_value

from dataclasses import asdict

def ee_traj(
    env: ManagerBasedRLEnv
):
    names = ["pos_w", "quat_w", "lin_vel_b", "ang_vel_b", "lin_acc_b", "ang_acc_b"]
    dec = env.cfg.decimation
    idx = 0
    #print(env.scene['ee_imu'].data)
    try:
        #print(env.scene['ee_imu'].data)
        data = asdict(env.scene['ee_imu'].data)
    except RuntimeError:
        return torch.zeros((env.num_envs, 19*env.cfg.decimation), device=env.device) # 7 pose, 6 vel, 6 acc
    
    for name in names:
        dim = 4 if "quat" in name else 3
        #print(data[name].size())
        env.traj_data[:, idx:idx+dec*dim] = data[name].view((env.num_envs, dec*dim))
        idx += dec * dim

    return env.traj_data


def fingertip_pos(
    env: ManagerBasedRLEnv
):
    try:
        compute_keypoint_value(env)#, dt=env.physics_dt))
        return env.fingertip_midpoint_pos
    except AttributeError: # obs is called before init so data structures not in place
        return torch.zeros((env.num_envs, 3), device=env.device)

def fingertip_quat(
    env: ManagerBasedRLEnv
):
    try:
        compute_keypoint_value(env)#, dt=env.physics_dt))
        return env.fingertip_midpoint_quat
    except AttributeError: # obs is called before init so data structures not in place
        return torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).unsqueeze(0).repeat(env.num_envs, 1)

def ee_linvel(
    env: ManagerBasedRLEnv
):
    try:
        compute_keypoint_value(env)#, dt=env.physics_dt))
        return env.ee_linvel_fd
    except AttributeError: # obs is called before init so data structures not in place
        return torch.zeros((env.num_envs, 3), device=env.device)

def ee_angvel(
    env: ManagerBasedRLEnv
):
    try:
        compute_keypoint_value(env)#, dt=env.physics_dt))
        return env.ee_angvel_fd
    except AttributeError: # obs is called before init so data structures not in place
        return torch.zeros((env.num_envs, 3), device=env.device)
    
def ee_linacc(
    env: ManagerBasedRLEnv
):
    try:
        compute_keypoint_value(env)#, dt=env.physics_dt))
        return env.ee_linacc_fd
    except AttributeError: # obs is called before init so data structures not in place
        return torch.zeros((env.num_envs, 3), device=env.device)

def ee_angacc(
    env: ManagerBasedRLEnv
):
    try:
        compute_keypoint_value(env)#, dt=env.physics_dt))
        return env.ee_angacc_fd
    except AttributeError: # obs is called before init so data structures not in place
        return torch.zeros((env.num_envs, 3), device=env.device)

def held_fixed_relative_pos(
    env: ManagerBasedRLEnv
):
    try:
        compute_keypoint_value(env)#, dt=env.physics_dt))
        return env.fingertip_midpoint_pos - env.fixed_pos
    except AttributeError:
        return torch.zeros((env.num_envs, 3), device=env.device)


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
        compute_keypoint_value(env)#, dt=env.physics_dt))
        #print("Held pose", env.target_held_base_pos.device, env.target_held_base_quat.device)
        return torch.cat([env.target_held_base_pos, env.target_held_base_quat], dim=1)
    except:
        return torch.zeros(env.num_envs, 7)

def fixed_asset_pose(
        env: ManagerBasedRLEnv
):
    try:
        compute_keypoint_value(env)#, dt=env.physics_dt))
        #print("Fixed Pose:", env.fixed_pos.device, env.fixed_quat.device)
        return torch.cat([env.fixed_pos, env.fixed_quat], dim=1)
    except:
        return torch.zeros(env.num_envs, 7)

def force_torque_sensor(
        env: ManagerBasedRLEnv
):
    try:
        return torch.tanh( 0.0011 * env.robot_av.get_measured_joint_forces()[:,8,:] )
    except:
        print("WARNING DO NOT HAVE FORCE TORQUE SENSING WORKING CORRECTLY")
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