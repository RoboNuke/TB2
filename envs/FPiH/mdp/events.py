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


from omni.isaac.core.articulations import ArticulationView

from omni.isaac.lab.utils.math import subtract_frame_transforms

import omni.isaac.core.utils.extensions as extensions_utils
extensions_utils.enable_extension("omni.isaac.robot_assembler")
from omni.isaac.robot_assembler import RobotAssembler,AssembledRobot
import numpy as np
"""
def randomize_peg(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    width_range: dict[str, tuple[float, float]],
    length_range: dict[str, tuple[float, float]]
):
    stage = omni.usd.get_context().get_stage()

    peg_paths = sim_utils.find_matching_prim_paths("{ENV_REGEX_NS}/Peg")
    # manually clone prims if the source prim path is a regex expression
    with Sdf.ChangeBlock():
        for prim_path in peg_paths:
            prim_spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), prim_path)

            width_spec = prim_spec.GetAttributeAtPath(prim_path + "/geometry/mesh.radius")
            height_spec = prim_spec.GetAttributeAtPath(prim_path + "/geometry/mesh.radius")
            length_spec = prim_spec.GetAttributeAtPath(prim_path + "/geometry/mesh.radius")
"""

def afix_peg_to_robot(
        env: ManagerBasedEnv,
        env_ids: torch.Tensor
):
    ra = RobotAssembler()

    #ra.assemble_rigid_bodies(
    #    "/World/envs/env_.*/Robot"
    #)

    """
    robot: Articulation = env.scene["robot"]
    hand_state = robot.data.body_link_state_w[:, 8, :]
    env.peg_init_pos = hand_state[:, :3]
    offset = torch.zeros_like(env.peg_init_pos[:,:3])
    offset[:,2] += 0.025
    env.peg_init_qua = hand_state[:, 3:7]
    env.peg_init_pos = env.peg_init_pos + quat_rotate(env.peg_init_qua, offset)
    print(hand_state[0,:,:])
    """
    for i in range(env.num_envs):
        ra.create_fixed_joint(
            prim_path = f"/World/envs/env_{i}/Robot/panda_hand",
            target0 = f"/World/envs/env_{i}/Robot/panda_hand",
            target1 = f"/World/envs/env_{i}/Peg",
            fixed_joint_offset = np.array([0,0, env.cfg.panda_ee_offset + 0.01])
        )

def init_ft_sensor(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    prim_paths_expr="/World/envs/env_.*/Robot"
):
    stage = omni.usd.get_context().get_stage()

    robot_paths = sim_utils.find_matching_prim_paths("/{ENV_REGEX_NS}/Robot")
    #print("init ft_sensor")
    #print(prim_paths_expr)
    env.scene.cfg.robot_av = ArticulationView(prim_paths_expr=prim_paths_expr) #, enable_dof_force_sensors=True)
    env.scene.cfg.robot_av.initialize()
    #print("ft sensor ready to go!")

def init_default_hole_pos(
        env: ManagerBasedEnv,
        env_ids: torch.Tensor
):
    # set hole box's default locations based on size
    holes = RigidObjectCollection = env.scene["hole"]

    n = len(env.cfg.hole_init_poses)
    for i in range(env.num_envs):
        for hole_idx in range(4):
            try:
                pos = env.cfg.hole_init_poses[i % n][hole_idx].pos
            except:
                pos = env.cfg.hole_init_poses[i % n][hole_idx]['pos']
            holes.data.default_object_state[i, hole_idx, 0] = pos[0] - env.cfg.x_offset
            holes.data.default_object_state[i, hole_idx, 1] = pos[1]
            holes.data.default_object_state[i, hole_idx, 2] = pos[2] 
        holes.data.default_object_state[i, 4, 0] = env.cfg.x_offset 
        holes.data.default_object_state[i, 4, 1] *= 0 
        holes.data.default_object_state[i, 4, 2] *= 0



def reset_hole(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]]#,
    #asset_cfg: SceneEntityCfg = SceneEntityCfg("hole"),
):
    asset: RigidObjectCollection = env.scene["hole"]

    # get default root state
    states = asset.data.default_object_state[env_ids].clone()
    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    rev_orient_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], -rand_samples[:, 5])
    R = math_utils.matrix_from_quat(orientations_delta)
    states = states.unsqueeze(-1)

    for i in range(states.shape[1]):
        states[:, i, 0:3] = torch.matmul(R, states[:, i, 0:3])
        states[:, i, 0:3, 0] += env.scene.env_origins[env_ids, :] + rand_samples[:, 0:3] 
        if i < 4:
            states[:, i, 3:7, 0] = math_utils.quat_mul(states[:, i, 3:7, 0], orientations_delta)
        else:
            states[:, i, 3:7, 0] = math_utils.quat_mul(states[:, i, 3:7, 0], rev_orient_delta)
    states = states.squeeze(-1)
    
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    for i in range(4):
        velocities = states[:, i, 7:13] + rand_samples

    #print(env.scene.extras)
    # set into the physics simulation
    asset.write_object_pose_to_sim(states[:,:,0:7], env_ids=env_ids)
    asset.write_object_velocity_to_sim(states[:,:,7:13], env_ids=env_ids)
    #print("wrote block pose")
    

def reset_peg(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor
):
    if not env.cfg.have_peg_pose:
        env.cfg.have_peg_pose = True
    #    for i in range(4):
        env.sim.step()
        ee_tf_data = env.scene["ee_frame"].data
        env.peg_init_pos = ee_tf_data.target_pos_w[env_ids, 0, :]
        offset = torch.zeros_like(env.peg_init_pos[:,:3])
        #print(offset.shape)
        #offset[:,0] -= .00625
        offset[:,2] += 0.025
        env.peg_init_qua = ee_tf_data.target_quat_w[env_ids, 0, :]
        env.peg_init_pos = env.peg_init_pos + quat_rotate(env.peg_init_qua, offset)
    #print(env.peg_init_qua[0,:], quat_rotate(env.peg_init_qua, offset)[0,:], env.peg_init_pos[0,:])
    #print(env.peg_init_pos.shape, env.peg_init_qua.shape, torch.cat([env.peg_init_pos, env.peg_init_qua], dim=-1).shape)

    # reset peg
    asset: RigidObject = env.scene["peg"]

    asset.write_root_state_to_sim(
        torch.cat([
                env.peg_init_pos[env_ids,:], 
                env.peg_init_qua[env_ids,:], 
                torch.zeros( (len(env_ids), 6), device=env.device)
            ], 
            dim=-1
        ), 
        env_ids=env_ids
    )

def reset_robot(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor
):
    default_joint_pos = env.scene['robot'].data.default_joint_pos[env_ids].clone()
    default_joint_pos[:,-2:] = 0.025
    
    default_joint_vel = env.scene['robot'].data.default_joint_vel[env_ids].clone()
    # set into the physics simulation
    env.scene['robot'].write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)
    #print("wrote peg pose")
    #help(asset)
    
def move_peg(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor
):
    ee_tf_data = env.scene["ee_frame"].data
    offset = torch.zeros_like(ee_tf_data.target_pos_w[:, 0, :3])
    offset[:,0] -= .00625
    offset[:,2] += 0.0255
    new_peg_q = ee_tf_data.target_quat_w[:,0,:]
    new_peg_pos = ee_tf_data.target_pos_w[:, 0, :]
    #print(new_peg_pos.shape, offset.shape)
    new_peg_pos = new_peg_pos + quat_rotate(new_peg_q, offset)
    asset: RigidObject = env.scene["peg"]

    asset.write_root_pose_to_sim(
        torch.cat([env.peg_init_pos[env_ids,:], env.peg_init_qua[env_ids,:]], dim=-1), 
        env_ids = env_ids
    )

