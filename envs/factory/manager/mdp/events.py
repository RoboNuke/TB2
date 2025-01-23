from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import carb
import omni.physics.tensors.impl.api as physx

import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.actuators import ImplicitActuator
from omni.isaac.lab.assets import Articulation, DeformableObject, RigidObject, RigidObjectCollection
from omni.isaac.lab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from omni.isaac.lab.terrains import TerrainImporter
from omni.isaac.lab.utils.math import quat_apply, quat_rotate, quat_rotate_inverse

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv

import omni.usd
import omni.kit.commands
from pxr import Usd, UsdGeom, Gf, Sdf


from envs.factory.manager import factory_control as fc
from omni.isaac.core.articulations import ArticulationView

from omni.isaac.lab.utils.math import subtract_frame_transforms

import omni.isaac.core.utils.extensions as extensions_utils
extensions_utils.enable_extension("omni.isaac.robot_assembler")
from omni.isaac.robot_assembler import RobotAssembler,AssembledRobot
import numpy as np

from envs.factory.manager.factory_manager_task_cfg import PegInsert, FactoryTask

import omni.isaac.core.utils.torch as torch_utils
from omni.isaac.lab.utils.math import axis_angle_from_quat

def set_body_inertias(
        env: ManagerBasedEnv,
        env_ids: torch.Tensor        
):
    """Note: this is to account for the asset_options.armature parameter in IGE."""
    env.fixed_asset_pos = [0.001, 0.001, 0.001]
    robot: Articulation = env.scene["robot"]
    inertias = robot.root_physx_view.get_inertias()
    offset = torch.zeros_like(inertias)
    offset[:, :, [0, 4, 8]] += 0.01
    new_inertias = inertias + offset
    robot.root_physx_view.set_inertias(new_inertias, torch.arange(env.num_envs))

def set_friction(env, asset, value):
    """Update material properties for a given asset."""
    materials = asset.root_physx_view.get_material_properties()
    materials[..., 0] = value  # Static friction.
    materials[..., 1] = value  # Dynamic friction.
    env_ids = torch.arange(env.scene.num_envs, device="cpu")
    asset.root_physx_view.set_material_properties(materials, env_ids)

def set_default_dynamics_parameters(
        env: ManagerBasedEnv,
        env_ids: torch.Tensor        
):
    """Set parameters defining dynamic interactions."""
    robot: Articulation = env.scene["robot"]
    """env.default_gains = torch.tensor(env.cfg.ctrl.default_task_prop_gains, device=env.device).repeat(
        (env.num_envs, 1)
    )

    env.pos_threshold = torch.tensor(env.cfg.ctrl.pos_action_threshold, device=env.device).repeat(
        (env.num_envs, 1)
    )
    env.rot_threshold = torch.tensor(env.cfg.ctrl.rot_action_threshold, device=env.device).repeat(
        (env.num_envs, 1)
    )"""

    # Set masses and frictions.
    held: Articulation = env.scene["held_asset"]
    fixed: Articulation = env.scene["fixed_asset"]
    set_friction(env, held, env.cfg_task.held_asset_cfg.friction)
    set_friction(env, fixed, env.cfg_task.fixed_asset_cfg.friction)
    set_friction(env, robot, env.cfg_task.robot_cfg.friction) 

    compute_keypoint_value(env, dt=env.physics_dt)

def init_tensors(
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        task_cfg: FactoryTask = PegInsert(), 
):
    
    """Initialize tensors once."""
    env.identity_quat = (
        torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).unsqueeze(0).repeat(env.num_envs, 1)
    )
    fixed_asset: Articulation = env.scene["fixed_asset"]

    env.fixed_pos = fixed_asset.data.root_link_pos_w - env.scene.env_origins
    env.fixed_quat = fixed_asset.data.root_link_quat_w

    env.cfg_task = task_cfg

    # get offsets of held object in local frame
    held_base_x_offset = 0.0
    if env.cfg_task.name == "peg_insert":
        held_base_z_offset = 0.0
    elif env.cfg_task.name == "gear_mesh":
        gear_base_offset = env._get_target_gear_base_offset()
        held_base_x_offset = gear_base_offset[0]
        held_base_z_offset = gear_base_offset[2]
    elif env.cfg_task.name == "nut_thread":
        held_base_z_offset = env.cfg_task.fixed_asset_cfg.base_height
    else:
        raise NotImplementedError("Task not implemented")
    
    # calculate local to held object frame, the target frame
    env.held_base_pos_local = torch.tensor([0.0, 0.0, 0.0], device=env.device).repeat((env.num_envs, 1))
    env.held_base_pos_local[:, 0] = held_base_x_offset
    env.held_base_pos_local[:, 2] = held_base_z_offset
    env.held_base_quat_local = env.identity_quat.clone().detach()

    env.held_base_pos = torch.zeros_like(env.held_base_pos_local)
    env.held_base_quat = torch.zeros_like(env.held_base_quat_local)

    num_keypoints = env.cfg_task.num_keypoints
    offsets = torch.zeros((num_keypoints, 3), device=env.device)
    offsets[:, -1] = torch.linspace(0.0, 1.0, num_keypoints, device=env.device) - 0.5
    env.keypoint_offsets = offsets * env.cfg_task.keypoint_scale
    env.keypoints_held = torch.zeros((env.num_envs, env.cfg_task.num_keypoints, 3), device=env.device)
    env.keypoints_fixed = torch.zeros_like(env.keypoints_held, device=env.device)

    # Used to compute target poses.
    env.fixed_success_pos_local = torch.zeros((env.num_envs, 3), device=env.device)
    if env.cfg_task.name == "peg_insert":
        env.fixed_success_pos_local[:, 2] = 0.0
    elif env.cfg_task.name == "gear_mesh":
        gear_base_offset = env._get_target_gear_base_offset()
        env.fixed_success_pos_local[:, 0] = gear_base_offset[0]
        env.fixed_success_pos_local[:, 2] = gear_base_offset[2]
    elif env.cfg_task.name == "nut_thread":
        head_height = env.cfg_task.fixed_asset_cfg.base_height
        shank_length = env.cfg_task.fixed_asset_cfg.height
        thread_pitch = env.cfg_task.fixed_asset_cfg.thread_pitch
        env.fixed_success_pos_local[:, 2] = head_height + shank_length - thread_pitch * 1.5
    else:
        raise NotImplementedError("Task not implemented")

    #define the target for the base frame
    env.target_held_base_pos = torch.zeros((env.num_envs, 3), device=env.device)
    env.target_held_base_quat = env.identity_quat.clone().detach()
    env.target_held_base_quat[:], env.target_held_base_pos[:] = torch_utils.tf_combine(
        env.fixed_quat, env.fixed_pos, env.identity_quat, env.fixed_success_pos_local
    )

    env.time_keypoint_update = env.sim._current_time
    #env.ep_succeeded = torch.zeros((env.num_envs,), dtype=torch.long, device=env.device)
    #env.ep_success_times = torch.zeros((env.num_envs,), dtype=torch.long, device=env.device)
    robot: Articulation = env.scene['robot']
    env.fingertip_body_idx = robot.body_names.index("panda_fingertip_centered")

def reset_held_asset(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor
):
    held_asset: Articulation = env.scene["held_asset"]
    robot: Articulation = env.scene['robot']
    env.fingertip_body_idx = robot.body_names.index("panda_fingertip_centered")
    
    fingertip_midpoint_pos = (
        robot.data.body_link_pos_w[:, env.fingertip_body_idx] - env.scene.env_origins
    )
    fingertip_midpoint_quat = robot.data.body_link_quat_w[:, env.fingertip_body_idx]

    flip_z_quat = torch.tensor([0.0, 0.0, 1.0, 0.0], device=env.device).unsqueeze(0).repeat(env.num_envs, 1)
    fingertip_flipped_quat, fingertip_flipped_pos = torch_utils.tf_combine(
        q1=fingertip_midpoint_quat,
        t1=fingertip_midpoint_pos,
        q2=flip_z_quat,
        t2=torch.zeros_like(fingertip_midpoint_pos),
    )

    # get default gripper in asset transform
    held_asset_relative_pos, held_asset_relative_quat = get_handheld_asset_relative_pose(env)
    asset_in_hand_quat, asset_in_hand_pos = torch_utils.tf_inverse(
        held_asset_relative_quat, held_asset_relative_pos
    )

    translated_held_asset_quat, translated_held_asset_pos = torch_utils.tf_combine(
        q1=fingertip_flipped_quat, t1=fingertip_flipped_pos, q2=asset_in_hand_quat, t2=asset_in_hand_pos
    )

    # Add asset in hand randomization
    rand_sample = torch.rand((env.num_envs, 3), dtype=torch.float32, device=env.device)*0.0
    env.held_asset_pos_noise = 2 * (rand_sample - 0.5)  # [-1, 1]
    if env.cfg_task.name == "gear_mesh":
        env.held_asset_pos_noise[:, 2] = -rand_sample[:, 2]  # [-1, 0]

    held_asset_pos_noise = torch.tensor(env.cfg_task.held_asset_pos_noise, device=env.device)
    env.held_asset_pos_noise = env.held_asset_pos_noise @ torch.diag(held_asset_pos_noise)
    translated_held_asset_quat, translated_held_asset_pos = torch_utils.tf_combine(
        q1=translated_held_asset_quat,
        t1=translated_held_asset_pos,
        q2=env.identity_quat,
        t2=env.held_asset_pos_noise,
    )

    held_state = held_asset.data.default_root_state.clone()
    held_state[:, 0:3] = translated_held_asset_pos + env.scene.env_origins
    held_state[:, 3:7] = translated_held_asset_quat
    held_state[:, 7:] = 0.0
    held_asset.write_root_link_pose_to_sim(held_state[:, 0:7])
    held_asset.write_root_com_velocity_to_sim(held_state[:, 7:])
    held_asset.reset()
    step_sim_no_action(env)

def reset_fixed_asset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor
):
    asset: RigidObjectCollection = env.scene["fixed_asset"]

    # get default root state
    states = asset.data.default_root_state[env_ids].clone()
    
    # poses
    pose_range = {}
    pose_range['x'] = (-env.cfg_task.fixed_asset_init_pos_noise[0], env.cfg_task.fixed_asset_init_pos_noise[0])
    pose_range['y'] = (-env.cfg_task.fixed_asset_init_pos_noise[1], env.cfg_task.fixed_asset_init_pos_noise[1])
    pose_range['yaw'] = (
        (env.cfg_task.fixed_asset_init_orn_deg - env.cfg_task.fixed_asset_init_orn_range_deg/2) * 3.14159 / 180.0, 
        (env.cfg_task.fixed_asset_init_orn_deg + env.cfg_task.fixed_asset_init_orn_range_deg/2) * 3.14159 / 180.0
    )
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    
    
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    states = states.unsqueeze(-1)

    #states[:, 0:3] = torch.matmul(R, states[:, 0:3])
    states[:, 0:3, 0] += rand_samples[:, 0:3]  + env.scene.env_origins[env_ids, :] 
    states[:, 3:7, 0] = math_utils.quat_mul(states[:, 3:7, 0], orientations_delta)
    states = states.squeeze(-1)
    
    velocity_range = {}
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    
    states[:, 7:13] += rand_samples

    #print(env.scene.extras)
    # set into the physics simulation
    asset.write_root_pose_to_sim(states[:,0:7], env_ids=env_ids)
    asset.write_root_velocity_to_sim(states[:,7:13], env_ids=env_ids)
    #print("wrote block pose")
    step_sim_no_action(env)

def set_assets_to_default_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor
):
    held_asset: Articulation = env.scene["held_asset"]
    fixed_asset: Articulation = env.scene["fixed_asset"]
    held_state = held_asset.data.default_root_state.clone()[env_ids]
    held_state[:, 0:3] += env.scene.env_origins[env_ids]
    held_state[:, 7:] = 0.0
    held_asset.write_root_link_pose_to_sim(held_state[:, 0:7], env_ids=env_ids)
    held_asset.write_root_com_velocity_to_sim(held_state[:, 7:], env_ids=env_ids)
    held_asset.reset()

    fixed_state = fixed_asset.data.default_root_state.clone()[env_ids]
    fixed_state[:, 0:3] += env.scene.env_origins[env_ids]
    fixed_state[:, 7:] = 0.0
    fixed_asset.write_root_link_pose_to_sim(fixed_state[:, 0:7], env_ids=env_ids)
    fixed_asset.write_root_com_velocity_to_sim(fixed_state[:, 7:], env_ids=env_ids)
    fixed_asset.reset()

def set_franka_to_default_pose(
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        #joints= [0.00871, -0.10368, -0.00794, -1.49139, -0.00083, 1.38774, 0.0]
        joints = [0.0, 3.14159/8, 0.0, -3.14159*5/8, 0.0, 3.14159*3/4, 3.14159/4]
):
    """Return Franka to its default joint position."""
    robot: Articulation = env.scene["robot"]
    gripper_width = env.cfg_task.held_asset_cfg.diameter / 2 * 1.25
    joint_pos = robot.data.default_joint_pos[env_ids]
    joint_pos[:, 7:] = gripper_width  # MIMIC
    joint_pos[:, :7] = torch.tensor(joints, device=env.device)[None, :]
    joint_vel = torch.zeros_like(joint_pos)
    joint_effort = torch.zeros_like(joint_pos)
    #print(f"Resetting {len(env_ids)} envs...")
    robot.set_joint_position_target(joint_pos, env_ids=env_ids)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    robot.reset()
    robot.set_joint_effort_target(joint_effort, env_ids=env_ids)

    step_sim_no_action(env)

def step_sim_no_action(env):
    env.scene.write_data_to_sim()
    env.sim.step(render=False)
    env.scene.update(dt=env.physics_dt)
    compute_keypoint_value(env, dt=env.physics_dt)

def get_handheld_asset_relative_pose(env):
    """Get default relative pose between help asset and fingertip."""
    if env.cfg_task.name == "peg_insert":
        held_asset_relative_pos = torch.tensor([0.0, 0.0, 0.0], device=env.device).repeat((env.num_envs, 1))
        held_asset_relative_pos[:, 2] = env.cfg_task.held_asset_cfg.height
        held_asset_relative_pos[:, 2] -= env.cfg_task.robot_cfg.franka_fingerpad_length
    elif env.cfg_task.name == "gear_mesh":
        held_asset_relative_pos = torch.tensor([0.0, 0.0, 0.0], device=env.device).repeat((env.num_envs, 1))
        gear_base_offset = env._get_target_gear_base_offset()
        held_asset_relative_pos[:, 0] += gear_base_offset[0]
        held_asset_relative_pos[:, 2] += gear_base_offset[2]
        held_asset_relative_pos[:, 2] += env.cfg_task.held_asset_cfg.height / 2.0 * 1.1
    elif env.cfg_task.name == "nut_thread":
        held_asset_relative_pos = torch.tensor([0.0, 0.0, 0.0], device=env.device).repeat((env.num_envs, 1))
    else:
        raise NotImplementedError("Task not implemented")

    held_asset_relative_quat = env.identity_quat
    if env.cfg_task.name == "nut_thread":
        # Rotate along z-axis of frame for default position.
        initial_rot_deg = env.cfg_task.held_asset_rot_init
        rot_yaw_euler = torch.tensor([0.0, 0.0, initial_rot_deg * np.pi / 180.0], device=env.device).repeat(
            env.num_envs, 1
        )
        held_asset_relative_quat = torch_utils.quat_from_euler_xyz(
            roll=rot_yaw_euler[:, 0], pitch=rot_yaw_euler[:, 1], yaw=rot_yaw_euler[:, 2]
        )

    return held_asset_relative_pos, held_asset_relative_quat

def compute_keypoint_value(
        env: ManagerBasedRLEnv,
        dt: float
):
    # ensure we don't repeat this calculation
    now = env.sim._current_time
    #print((now - env.time_keypoint_update), env.sim.cfg.dt * (env.cfg.decimation-1))
    if (now - env.time_keypoint_update) < env.sim.cfg.dt * (env.cfg.decimation-1):#todo decimation
        return
    #print("Calculating")
    env.time_keypoint_update = now
    
    held_asset: Articulation = env.scene["held_asset"]
    fixed_asset: Articulation = env.scene['fixed_asset']

    env.held_pos = held_asset.data.root_link_pos_w - env.scene.env_origins
    env.held_quat = held_asset.data.root_link_quat_w

    env.fixed_pos = fixed_asset.data.root_link_pos_w - env.scene.env_origins
    env.fixed_quat = fixed_asset.data.root_link_quat_w

    # update keypoint location
    env.held_base_quat[:], env.held_base_pos[:] = torch_utils.tf_combine(
        env.held_quat, env.held_pos, env.held_base_quat_local, env.held_base_pos_local
    )
    env.target_held_base_quat[:], env.target_held_base_pos[:] = torch_utils.tf_combine(
        env.fixed_quat, env.fixed_pos, env.identity_quat, env.fixed_success_pos_local
    )

    # Compute pos of keypoints on held asset in world frame
    for idx, keypoint_offset in enumerate(env.keypoint_offsets):
        env.keypoints_held[:, idx] = torch_utils.tf_combine(
            env.held_base_quat, 
            env.held_base_pos, 
            env.identity_quat, 
            keypoint_offset.repeat(env.num_envs, 1)
        )[1]
        env.keypoints_fixed[:, idx] = torch_utils.tf_combine(
            env.target_held_base_quat,
            env.target_held_base_pos,
            env.identity_quat,
            keypoint_offset.repeat(env.num_envs, 1),
        )[1]

    env.keypoint_dist = torch.norm(env.keypoints_held - env.keypoints_fixed, p=2, dim=-1).mean(-1)
    #print("keypoint dist:", env.keypoint_dist)

def init_ft_sensor(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    prim_paths_expr="/World/envs/env_.*/Robot"
):
    stage = omni.usd.get_context().get_stage()

    robot_paths = sim_utils.find_matching_prim_paths("/{ENV_REGEX_NS}/Robot")
    #print("init ft_sensor")
    #print(prim_paths_expr)
    env.robot_av = ArticulationView(prim_paths_expr=prim_paths_expr) #, enable_dof_force_sensors=True)
    env.robot_av.initialize()
    #print("ft sensor ready to go!")