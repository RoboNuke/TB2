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

    compute_intermediate_values(env, dt=env.physics_dt)

def init_tensors(
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        task_cfg: FactoryTask = PegInsert(), 
):
    
    """Initialize tensors once."""
    env.identity_quat = (
        torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).unsqueeze(0).repeat(env.num_envs, 1)
    )

    env.cfg_task = task_cfg
    robot: Articulation = env.scene["robot"]
    # Control targets.
    env.ctrl_target_joint_pos = torch.zeros((env.num_envs, robot.num_joints), device=env.device)
    env.ctrl_target_fingertip_midpoint_pos = torch.zeros((env.num_envs, 3), device=env.device)
    env.ctrl_target_fingertip_midpoint_quat = torch.zeros((env.num_envs, 4), device=env.device)

    # Fixed asset.
    env.fixed_pos_action_frame = torch.zeros((env.num_envs, 3), device=env.device)
    env.fixed_pos_obs_frame = torch.zeros((env.num_envs, 3), device=env.device)
    env.init_fixed_pos_obs_noise = torch.zeros((env.num_envs, 3), device=env.device)

    # Held asset
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

    env.held_base_pos_local = torch.tensor([0.0, 0.0, 0.0], device=env.device).repeat((env.num_envs, 1))
    env.held_base_pos_local[:, 0] = held_base_x_offset
    env.held_base_pos_local[:, 2] = held_base_z_offset
    env.held_base_quat_local = env.identity_quat.clone().detach()

    env.held_base_pos = torch.zeros_like(env.held_base_pos_local)
    env.held_base_quat = env.identity_quat.clone().detach()

    # Computer body indices.
    env.left_finger_body_idx = robot.body_names.index("panda_leftfinger")
    env.right_finger_body_idx = robot.body_names.index("panda_rightfinger")
    env.fingertip_body_idx = robot.body_names.index("panda_fingertip_centered")

    # Tensors for finite-differencing.
    env.last_update_timestamp = 0.0  # Note: This is for finite differencing body velocities.
    env.prev_fingertip_pos = torch.zeros((env.num_envs, 3), device=env.device)
    env.prev_fingertip_quat = env.identity_quat.clone()
    env.prev_joint_pos = torch.zeros((env.num_envs, 7), device=env.device)

    # Keypoint tensors.
    env.target_held_base_pos = torch.zeros((env.num_envs, 3), device=env.device)
    env.target_held_base_quat = env.identity_quat.clone().detach()

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

    env.ep_succeeded = torch.zeros((env.num_envs,), dtype=torch.long, device=env.device)
    env.ep_success_times = torch.zeros((env.num_envs,), dtype=torch.long, device=env.device)


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
        joints= [0.00871, -0.10368, -0.00794, -1.49139, -0.00083, 1.38774, 0.0]
):
    """Return Franka to its default joint position."""
    robot: Articulation = env.scene["robot"]
    gripper_width = env.cfg_task.held_asset_cfg.diameter / 2 * 1.25
    joint_pos = robot.data.default_joint_pos[env_ids]
    joint_pos[:, 7:] = gripper_width  # MIMIC
    joint_pos[:, :7] = torch.tensor(joints, device=env.device)[None, :]
    joint_vel = torch.zeros_like(joint_pos)
    joint_effort = torch.zeros_like(joint_pos)
    env.ctrl_target_joint_pos[env_ids, :] = joint_pos
    #print(f"Resetting {len(env_ids)} envs...")
    robot.set_joint_position_target(env.ctrl_target_joint_pos[env_ids], env_ids=env_ids)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    robot.reset()
    robot.set_joint_effort_target(joint_effort, env_ids=env_ids)

    step_sim_no_action(env)

def step_sim_no_action(env):
    env.scene.write_data_to_sim()
    env.sim.step(render=False)
    env.scene.update(dt=env.physics_dt)
    compute_intermediate_values(env, dt=env.physics_dt)


def randomize_init_state(
        env: ManagerBasedEnv,
        env_ids: torch.Tensor
):
    env.ep_succeeded[env_ids] = 0
    robot: Articulation = env.scene["robot"]
    held_asset: Articulation = env.scene["held_asset"]
    fixed_asset: Articulation = env.scene["fixed_asset"]

    # Disable gravity.
    physics_sim_view = sim_utils.SimulationContext.instance().physics_sim_view
    physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, 0.0))

    # (1.) Randomize fixed asset pose.
    fixed_state = fixed_asset.data.default_root_state.clone()[env_ids]
    # (1.a.) Position
    rand_sample = torch.rand((len(env_ids), 3), dtype=torch.float32, device=env.device)
    fixed_pos_init_rand = 2 * (rand_sample - 0.5)  # [-1, 1]
    fixed_asset_init_pos_rand = torch.tensor(
        env.cfg_task.fixed_asset_init_pos_noise, dtype=torch.float32, device=env.device
    )
    fixed_pos_init_rand = fixed_pos_init_rand @ torch.diag(fixed_asset_init_pos_rand)
    fixed_state[:, 0:3] += fixed_pos_init_rand + env.scene.env_origins[env_ids]
    # (1.b.) Orientation
    fixed_orn_init_yaw = np.deg2rad(env.cfg_task.fixed_asset_init_orn_deg)
    fixed_orn_yaw_range = np.deg2rad(env.cfg_task.fixed_asset_init_orn_range_deg)
    rand_sample = torch.rand((len(env_ids), 3), dtype=torch.float32, device=env.device)
    fixed_orn_euler = fixed_orn_init_yaw + fixed_orn_yaw_range * rand_sample
    fixed_orn_euler[:, 0:2] = 0.0  # Only change yaw.
    fixed_orn_quat = torch_utils.quat_from_euler_xyz(
        fixed_orn_euler[:, 0], fixed_orn_euler[:, 1], fixed_orn_euler[:, 2]
    )
    fixed_state[:, 3:7] = fixed_orn_quat
    # (1.c.) Velocity
    fixed_state[:, 7:] = 0.0  # vel
    # (1.d.) Update values.
    fixed_asset.write_root_link_pose_to_sim(fixed_state[:, 0:7], env_ids=env_ids)
    fixed_asset.write_root_com_velocity_to_sim(fixed_state[:, 7:], env_ids=env_ids)
    fixed_asset.reset()

    # (1.e.) Noisy position observation.
    fixed_asset_pos_noise = torch.randn((len(env_ids), 3), dtype=torch.float32, device=env.device)
    
    fixed_asset_pos_rand = torch.tensor(env.fixed_asset_pos, dtype=torch.float32, device=env.device)
    fixed_asset_pos_noise = fixed_asset_pos_noise @ torch.diag(fixed_asset_pos_rand)
    env.init_fixed_pos_obs_noise[:] = fixed_asset_pos_noise

    step_sim_no_action(env)

    # Compute the frame on the bolt that would be used as observation: fixed_pos_obs_frame
    # For example, the tip of the bolt can be used as the observation frame
    
    fixed_tip_pos_local = torch.zeros_like(env.fixed_pos)
    fixed_tip_pos_local[:, 2] += env.cfg_task.fixed_asset_cfg.height
    fixed_tip_pos_local[:, 2] += env.cfg_task.fixed_asset_cfg.base_height
    #if env.cfg_task.name == "gear_mesh":
    #    fixed_tip_pos_local[:, 0] = env._get_target_gear_base_offset()[0]

    _, fixed_tip_pos = torch_utils.tf_combine(
        env.fixed_quat, env.fixed_pos, env.identity_quat, fixed_tip_pos_local
    )
    env.fixed_pos_obs_frame[:] = fixed_tip_pos

    # (2) Move gripper to randomizes location above fixed asset. Keep trying until IK succeeds.
    # (a) get position vector to target
    bad_envs = env_ids.clone()
    ik_attempt = 0

    hand_down_quat = torch.zeros((env.num_envs, 4), dtype=torch.float32, device=env.device)
    env.hand_down_euler = torch.zeros((env.num_envs, 3), dtype=torch.float32, device=env.device)
    while True:
        n_bad = bad_envs.shape[0]

        above_fixed_pos = fixed_tip_pos.clone()
        above_fixed_pos[:, 2] += env.cfg_task.hand_init_pos[2]

        rand_sample = torch.rand((n_bad, 3), dtype=torch.float32, device=env.device)
        above_fixed_pos_rand = 2 * (rand_sample - 0.5)  # [-1, 1]
        hand_init_pos_rand = torch.tensor(env.cfg_task.hand_init_pos_noise, device=env.device)
        above_fixed_pos_rand = above_fixed_pos_rand @ torch.diag(hand_init_pos_rand)
        above_fixed_pos[bad_envs] += above_fixed_pos_rand

        # (b) get random orientation facing down
        hand_down_euler = (
            torch.tensor(env.cfg_task.hand_init_orn, device=env.device).unsqueeze(0).repeat(n_bad, 1)
        )

        rand_sample = torch.rand((n_bad, 3), dtype=torch.float32, device=env.device)
        above_fixed_orn_noise = 2 * (rand_sample - 0.5)  # [-1, 1]
        hand_init_orn_rand = torch.tensor(env.cfg_task.hand_init_orn_noise, device=env.device)
        above_fixed_orn_noise = above_fixed_orn_noise @ torch.diag(hand_init_orn_rand)
        hand_down_euler += above_fixed_orn_noise
        env.hand_down_euler[bad_envs, ...] = hand_down_euler
        hand_down_quat[bad_envs, :] = torch_utils.quat_from_euler_xyz(
            roll=hand_down_euler[:, 0], pitch=hand_down_euler[:, 1], yaw=hand_down_euler[:, 2]
        )

        # (c) iterative IK Method
        env.ctrl_target_fingertip_midpoint_pos[bad_envs, ...] = above_fixed_pos[bad_envs, ...]
        env.ctrl_target_fingertip_midpoint_quat[bad_envs, ...] = hand_down_quat[bad_envs, :]

        
        pos_error, aa_error = set_pos_inverse_kinematics(env, env_ids=bad_envs)
        pos_error = torch.linalg.norm(pos_error, dim=1) > 1e-3
        angle_error = torch.norm(aa_error, dim=1) > 1e-3
        any_error = torch.logical_or(pos_error, angle_error)
        bad_envs = bad_envs[any_error.nonzero(as_tuple=False).squeeze(-1)]

        # Check IK succeeded for all envs, otherwise try again for those envs
        if bad_envs.shape[0] == 0:
            break

        set_franka_to_default_pose(
            env, 
            env_ids=bad_envs,
            joints=[0.00871, -0.10368, -0.00794, -1.49139, -0.00083, 1.38774, 0.0]
        )

        ik_attempt += 1
        #print(f"IK Attempt: {ik_attempt}\tBad Envs: {bad_envs.shape[0]}")

    step_sim_no_action(env)

    # Add flanking gears after servo (so arm doesn't move them).
    """
    if env.cfg_task.name == "gear_mesh" and env.cfg_task.add_flanking_gears:
        small_gear_state = env._small_gear_asset.data.default_root_state.clone()[env_ids]
        small_gear_state[:, 0:7] = fixed_state[:, 0:7]
        small_gear_state[:, 7:] = 0.0  # vel
        env._small_gear_asset.write_root_link_pose_to_sim(small_gear_state[:, 0:7], env_ids=env_ids)
        env._small_gear_asset.write_root_com_velocity_to_sim(small_gear_state[:, 7:], env_ids=env_ids)
        env._small_gear_asset.reset()

        large_gear_state = env._large_gear_asset.data.default_root_state.clone()[env_ids]
        large_gear_state[:, 0:7] = fixed_state[:, 0:7]
        large_gear_state[:, 7:] = 0.0  # vel
        env._large_gear_asset.write_root_link_pose_to_sim(large_gear_state[:, 0:7], env_ids=env_ids)
        env._large_gear_asset.write_root_com_velocity_to_sim(large_gear_state[:, 7:], env_ids=env_ids)
        env._large_gear_asset.reset()
    """
    # (3) Randomize asset-in-gripper location.
    # flip gripper z orientation
    flip_z_quat = torch.tensor([0.0, 0.0, 1.0, 0.0], device=env.device).unsqueeze(0).repeat(env.num_envs, 1)
    fingertip_flipped_quat, fingertip_flipped_pos = torch_utils.tf_combine(
        q1=env.fingertip_midpoint_quat,
        t1=env.fingertip_midpoint_pos,
        q2=flip_z_quat,
        t2=torch.zeros_like(env.fingertip_midpoint_pos),
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
    rand_sample = torch.rand((env.num_envs, 3), dtype=torch.float32, device=env.device)
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

    #  Close hand
    # Set gains to use for quick resets.
    # TODO
    """reset_task_prop_gains = torch.tensor(env.cfg.ctrl.reset_task_prop_gains, device=env.device).repeat(
        (env.num_envs, 1)
    )
    reset_rot_deriv_scale = env.cfg.ctrl.reset_rot_deriv_scale
    env._set_gains(reset_task_prop_gains, reset_rot_deriv_scale)
    """

    step_sim_no_action(env)

    grasp_time = 0.0
    while grasp_time < 0.25:
        env.ctrl_target_joint_pos[env_ids, 7:] = 0.0  # Close gripper.
        env.ctrl_target_gripper_dof_pos = 0.0
        #TODO
        #env.close_gripper_in_place()
        step_sim_no_action(env)
        grasp_time += env.sim.get_physics_dt()

    #TODO
    env.prev_joint_pos = env.joint_pos[:, 0:7].clone()
    env.prev_fingertip_pos = env.fingertip_midpoint_pos.clone()
    env.prev_fingertip_quat = env.fingertip_midpoint_quat.clone()

    # Set initial actions to involve no-movement. Needed for EMA/correct penalties.
    #TODO
    #env.actions = torch.zeros_like(env.actions)
    #env.prev_actions = torch.zeros_like(env.actions)
    # Back out what actions should be for initial state.
    # Relative position to bolt tip.
    """env.fixed_pos_action_frame[:] = env.fixed_pos_obs_frame + env.init_fixed_pos_obs_noise

    pos_actions = env.fingertip_midpoint_pos - env.fixed_pos_action_frame
    pos_action_bounds = torch.tensor(env.cfg.ctrl.pos_action_bounds, device=env.device)
    pos_actions = pos_actions @ torch.diag(1.0 / pos_action_bounds)
    env.actions[:, 0:3] = env.prev_actions[:, 0:3] = pos_actions
    
    # Relative yaw to bolt.
    unrot_180_euler = torch.tensor([-np.pi, 0.0, 0.0], device=env.device).repeat(env.num_envs, 1)
    unrot_quat = torch_utils.quat_from_euler_xyz(
        roll=unrot_180_euler[:, 0], pitch=unrot_180_euler[:, 1], yaw=unrot_180_euler[:, 2]
    )

    fingertip_quat_rel_bolt = torch_utils.quat_mul(unrot_quat, env.fingertip_midpoint_quat)
    fingertip_yaw_bolt = torch_utils.get_euler_xyz(fingertip_quat_rel_bolt)[-1]
    fingertip_yaw_bolt = torch.where(
        fingertip_yaw_bolt > torch.pi / 2, fingertip_yaw_bolt - 2 * torch.pi, fingertip_yaw_bolt
    )
    fingertip_yaw_bolt = torch.where(
        fingertip_yaw_bolt < -torch.pi, fingertip_yaw_bolt + 2 * torch.pi, fingertip_yaw_bolt
    )

    yaw_action = (fingertip_yaw_bolt + np.deg2rad(180.0)) / np.deg2rad(270.0) * 2.0 - 1.0
    env.actions[:, 5] = env.prev_actions[:, 5] = yaw_action
    """
    # Zero initial velocity.
    env.ee_angvel_fd[:, :] = 0.0
    env.ee_linvel_fd[:, :] = 0.0

    # Set initial gains for the episode.
    #ACTION env._set_gains(env.default_gains)
    
    physics_sim_view.set_gravity(carb.Float3(*env.cfg.sim.gravity))


def get_handheld_asset_relative_pose(env):
    """Get default relative pose between help asset and fingertip."""
    if env.cfg_task.name == "peg_insert":
        held_asset_relative_pos = torch.zeros_like(env.held_base_pos_local)
        held_asset_relative_pos[:, 2] = env.cfg_task.held_asset_cfg.height
        held_asset_relative_pos[:, 2] -= env.cfg_task.robot_cfg.franka_fingerpad_length
    elif env.cfg_task.name == "gear_mesh":
        held_asset_relative_pos = torch.zeros_like(env.held_base_pos_local)
        gear_base_offset = env._get_target_gear_base_offset()
        held_asset_relative_pos[:, 0] += gear_base_offset[0]
        held_asset_relative_pos[:, 2] += gear_base_offset[2]
        held_asset_relative_pos[:, 2] += env.cfg_task.held_asset_cfg.height / 2.0 * 1.1
    elif env.cfg_task.name == "nut_thread":
        held_asset_relative_pos = env.held_base_pos_local
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

def compute_intermediate_values(env: ManagerBasedRLEnv, dt: float):
    """Get values computed from raw tensors. This includes adding noise."""
    # TODO: Check so we don't rerun this every step
    
    try:
        if env.iter_step == env.sim._current_time: #.common_step_counter:
            return
        else:
            env.iter_step = env.sim._current_time
    except AttributeError:
        env.iter_step = env.sim._current_time


    robot: Articulation = env.scene["robot"]
    held_asset: Articulation = env.scene["held_asset"]
    fixed_asset: Articulation = env.scene["fixed_asset"]
    env.fixed_pos = fixed_asset.data.root_link_pos_w - env.scene.env_origins
    env.fixed_quat = fixed_asset.data.root_link_quat_w

    env.held_pos = held_asset.data.root_link_pos_w - env.scene.env_origins
    env.held_quat = held_asset.data.root_link_quat_w

    env.fingertip_midpoint_pos = (
        robot.data.body_link_pos_w[:, env.fingertip_body_idx] - env.scene.env_origins
    )
    env.fingertip_midpoint_quat = robot.data.body_link_quat_w[:, env.fingertip_body_idx]
    env.fingertip_midpoint_linvel = robot.data.body_com_lin_vel_w[:, env.fingertip_body_idx]
    env.fingertip_midpoint_angvel = robot.data.body_com_ang_vel_w[:, env.fingertip_body_idx]

    jacobians = robot.root_physx_view.get_jacobians()

    env.left_finger_jacobian = jacobians[:, env.left_finger_body_idx - 1, 0:6, 0:7]
    env.right_finger_jacobian = jacobians[:, env.right_finger_body_idx - 1, 0:6, 0:7]
    env.fingertip_midpoint_jacobian = (env.left_finger_jacobian + env.right_finger_jacobian) * 0.5
    env.arm_mass_matrix = robot.root_physx_view.get_mass_matrices()[:, 0:7, 0:7]
    env.joint_pos = robot.data.joint_pos.clone()
    env.joint_vel = robot.data.joint_vel.clone()

    # Finite-differencing results in more reliable velocity estimates.
    env.ee_linvel_fd = (env.fingertip_midpoint_pos - env.prev_fingertip_pos) / dt
    env.prev_fingertip_pos = env.fingertip_midpoint_pos.clone()

    # Add state differences if velocity isn't being added.
    rot_diff_quat = torch_utils.quat_mul(
        env.fingertip_midpoint_quat, torch_utils.quat_conjugate(env.prev_fingertip_quat)
    )
    rot_diff_quat *= torch.sign(rot_diff_quat[:, 0]).unsqueeze(-1)
    rot_diff_aa = axis_angle_from_quat(rot_diff_quat)
    env.ee_angvel_fd = rot_diff_aa / dt
    env.prev_fingertip_quat = env.fingertip_midpoint_quat.clone()

    joint_diff = env.joint_pos[:, 0:7] - env.prev_joint_pos
    env.joint_vel_fd = joint_diff / dt
    env.prev_joint_pos = env.joint_pos[:, 0:7].clone()

    # Keypoint tensors.
    env.held_base_quat[:], env.held_base_pos[:] = torch_utils.tf_combine(
        env.held_quat, env.held_pos, env.held_base_quat_local, env.held_base_pos_local
    )
    env.target_held_base_quat[:], env.target_held_base_pos[:] = torch_utils.tf_combine(
        env.fixed_quat, env.fixed_pos, env.identity_quat, env.fixed_success_pos_local
    )

    # Compute pos of keypoints on held asset, and fixed asset in world frame
    for idx, keypoint_offset in enumerate(env.keypoint_offsets):
        env.keypoints_held[:, idx] = torch_utils.tf_combine(
            env.held_base_quat, env.held_base_pos, env.identity_quat, keypoint_offset.repeat(env.num_envs, 1)
        )[1]
        env.keypoints_fixed[:, idx] = torch_utils.tf_combine(
            env.target_held_base_quat,
            env.target_held_base_pos,
            env.identity_quat,
            keypoint_offset.repeat(env.num_envs, 1),
        )[1]

    env.keypoint_dist = torch.norm(env.keypoints_held - env.keypoints_fixed, p=2, dim=-1).mean(-1)
    env.last_update_timestamp = robot._data._sim_timestamp


def set_pos_inverse_kinematics(env, env_ids):
    """Set robot joint position using DLS IK."""
    ik_time = 0.0
    while ik_time < 0.25:
        # Compute error to target.
        pos_error, axis_angle_error = fc.get_pose_error(
            fingertip_midpoint_pos=env.fingertip_midpoint_pos[env_ids],
            fingertip_midpoint_quat=env.fingertip_midpoint_quat[env_ids],
            ctrl_target_fingertip_midpoint_pos=env.ctrl_target_fingertip_midpoint_pos[env_ids],
            ctrl_target_fingertip_midpoint_quat=env.ctrl_target_fingertip_midpoint_quat[env_ids],
            jacobian_type="geometric",
            rot_error_type="axis_angle",
        )

        delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)

        # Solve DLS problem.
        delta_dof_pos = fc._get_delta_dof_pos(
            delta_pose=delta_hand_pose,
            ik_method="dls",
            jacobian=env.fingertip_midpoint_jacobian[env_ids],
            device=env.device,
        )
        env.joint_pos[env_ids, 0:7] += delta_dof_pos[:, 0:7]
        env.joint_vel[env_ids, :] = torch.zeros_like(env.joint_pos[env_ids,])

        env.ctrl_target_joint_pos[env_ids, 0:7] = env.joint_pos[env_ids, 0:7]
        # Update dof state.

        robot: Articulation = env.scene["robot"]
        robot.write_joint_state_to_sim(env.joint_pos, env.joint_vel)
        robot.set_joint_position_target(env.ctrl_target_joint_pos)

        # Simulate and update tensors.
        step_sim_no_action(env)
        ik_time += env.physics_dt

    return pos_error, axis_angle_error

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


def close_gripper_in_place(self):
    """Keep gripper in current position as gripper closes."""
    actions = torch.zeros((self.num_envs, 6), device=self.device)
    ctrl_target_gripper_dof_pos = 0.0

    # Interpret actions as target pos displacements and set pos target
    pos_actions = actions[:, 0:3] * self.pos_threshold
    self.ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_actions

    # Interpret actions as target rot (axis-angle) displacements
    rot_actions = actions[:, 3:6]

    # Convert to quat and set rot target
    angle = torch.norm(rot_actions, p=2, dim=-1)
    axis = rot_actions / angle.unsqueeze(-1)

    rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)

    rot_actions_quat = torch.where(
        angle.unsqueeze(-1).repeat(1, 4) > 1.0e-6,
        rot_actions_quat,
        torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1),
    )
    self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(rot_actions_quat, self.fingertip_midpoint_quat)

    target_euler_xyz = torch.stack(torch_utils.get_euler_xyz(self.ctrl_target_fingertip_midpoint_quat), dim=1)
    target_euler_xyz[:, 0] = 3.14159
    target_euler_xyz[:, 1] = 0.0

    self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
        roll=target_euler_xyz[:, 0], pitch=target_euler_xyz[:, 1], yaw=target_euler_xyz[:, 2]
    )

    self.ctrl_target_gripper_dof_pos = ctrl_target_gripper_dof_pos
    self.generate_ctrl_signals()