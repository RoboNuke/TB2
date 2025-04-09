from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import subtract_frame_transforms, quat_apply
from omni.isaac.core.articulations import ArticulationView

import omni.isaac.core.utils.torch as torch_utils

from omni.isaac.lab.envs import mdp
import cv2

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

from envs.factory.manager.mdp.events import compute_keypoint_value

from dataclasses import asdict

def ee_traj(
    env: ManagerBasedRLEnv,
    relative: bool = False
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
        if relative and name == "pos_w":
            env.traj_data[:, idx:idx+dec*dim] = (data[name] - data[name][:,0,:]).view((env.num_envs, dec*dim))
        elif relative and name == "quat_w":
            env.traj_data[:, idx:idx+dec*dim] = torch_utils.quat_mul(
                data[name], torch_utils.quat_conjugate(data[name][:,0,:])
            ).view((env.num_envs, dec*dim))
        else:
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

def robot_fixed_relative_pos(
    env: ManagerBasedRLEnv,
    imu_relative: bool = False
):
    try:
        compute_keypoint_value(env)#, dt=env.physics_dt))
        if imu_relative:
            return env.fixed_pos - env.scene['ee_imu'].data.pos_w[:,0,:]
        else:
            return env.fixed_pos - env.fingertip_midpoint_pos
    except AttributeError:
        return torch.zeros((env.num_envs, 3), device=env.device)

def robot_fixed_relative_quat(
    env: ManagerBasedRLEnv,
    imu_relative: bool = False
):
    try: 
        compute_keypoint_value(env)
        if imu_relative:
            return torch_utils.quat_mul(
                    env.target_held_base_quat, torch_utils.quat_conjugate(env.scene['ee_imu'].data.quat_w[:,0,:])
                )
        else:
            return torch_utils.quat_mul(
                    env.target_held_base_quat, torch_utils.quat_conjugate(env.fingertip_midpoint_quat)
                )
    except AttributeError:
        return torch.zeros((env.num_envs, 4), device=env.device)

def robot_held_relative_pos(
    env: ManagerBasedRLEnv,
    imu_relative: bool = False
):
    try:
        compute_keypoint_value(env)#, dt=env.physics_dt))
        if imu_relative:
            return env.held_pos - env.scene['ee_imu'].data.pos_w[:,0,:]
        else:
            return env.held_pos - env.fingertip_midpoint_pos
    except AttributeError:
        return torch.zeros((env.num_envs, 3), device=env.device)

def robot_held_relative_quat(
    env: ManagerBasedRLEnv,
    imu_relative: bool = False
):
    try: 
        compute_keypoint_value(env)
        if imu_relative:
            return torch_utils.quat_mul(
                    env.held_quat, torch_utils.quat_conjugate(env.scene['ee_imu'].data.quat_w[:,0,:])
                )
        else:
            return torch_utils.quat_mul(
                    env.held_quat, torch_utils.quat_conjugate(env.fingertip_midpoint_quat)
                )
    except AttributeError:
        return torch.zeros((env.num_envs, 4), device=env.device)
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
    return env.scene['force_torque_sensor'].data.net_forces_w_history

def old_force_torque_sensor(
        env: ManagerBasedRLEnv
):
    try:
        return torch.tanh( 0.0011 * env.robot_av.get_measured_joint_forces()[:,8,:] )
    except:
        print("WARNING DO NOT HAVE FORCE TORQUE SENSING WORKING CORRECTLY")
        return torch.zeros((env.num_envs, 6), dtype=torch.float32, device=env.device)
    

def force_torque_sensor_scaled(
        env: ManagerBasedRLEnv
):
    dec = env.scene['force_torque_sensor'].history_length
    #print(env.scene['force_torque_sensor'].data.net_forces_w_history[0,:,:])
    ft_data = env.scene['force_torque_sensor'].data.net_forces_w_history.view((env.num_envs, dec*6))
    return torch.tanh( 0.001 * ft_data)

def peg_contact_sensor(
    env: ManagerBasedRLEnv
):
    return torch.linalg.norm(env.scene['peg_contact_force'].data.net_forces_w[:,0,:], dim=1)

def camera_image(
    env: ManagerBasedRLEnv
):
    import matplotlib.pyplot as plt
    import torchvision
    import torchvision.transforms.functional as F
    #print("cam img:", env.scene.keys())
    #print("recording:", env.cfg.recording)
    if env.cfg.recording:
        cam_data = env.scene['tiled_camera'].data
        img = cam_data.output['rgb']
        img = img.permute(0,3,1,2)
        #print(img.size())
        image_grid = torchvision.utils.make_grid(img)
        #print(image_grid.size())
        #plt.imshow(F.to_pil_image(image_grid))
        #plt.savefig(f"imgs/reset_img_{env.common_step_counter}.png", dpi=450)
        #plt.show()

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