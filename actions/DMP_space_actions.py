from __future__ import annotations

import torch
from collections.abc import Sequence
#from typing import TYPE_CHECKING

import omni.log
#from pxr import UsdPhysics

import omni.isaac.lab.utils.math as math_utils
import omni.isaac.lab.utils.string as string_utils
from omni.isaac.lab.assets.articulation import Articulation
from omni.isaac.lab.controllers.differential_ik import DifferentialIKController
#from omni.isaac.lab.controllers.operational_space import OperationalSpaceController
from omni.isaac.lab.managers.action_manager import ActionTerm
#from omni.isaac.lab.sensors import ContactSensor, ContactSensorCfg, FrameTransformer, FrameTransformerCfg
#from omni.isaac.lab.sim.utils import find_matching_prims
import actions.DMP_space_actions_cfg as dmp_actions#import DMPDiffIKActionCfg
from wrappers.DMP.discrete_dmp import DiscreteDMP
from wrappers.DMP.cs import CS

class DMPDiffIKAction(ActionTerm):
    r"""DMP Inverse Kinematics action term.

    This action term performs pre-processing of the weights for a DMP, scaling them appropriately, and then
    determines EE goal for entire trajectory.

    .. math::
        \text{action} = \text{scaling} \times \text{input action}
        \text{joint position} = J^{-} \times \text{action}

    where :math:`\text{scaling}` is the scaling applied to the input action, and :math:`\text{input action}`
    is the input action from the user, :math:`J` is the Jacobian over the articulation's actuated joints,
    and \text{joint position} is the desired joint position command for the articulation's joints.
    """

    cfg: dmp_actions.DMPDiffIKActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _scale: torch.Tensor
    """The scaling factor applied to the input action. Shape is (1, action_dim)."""
    _clip: torch.Tensor
    """The clip applied to the input action."""
    pos_dmp: DiscreteDMP
    """The position dmp"""

    def __init__(self, cfg: dmp_actions.DMPDiffIKActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        # resolve the joints over which the action term is applied
        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names)
        self._num_joints = len(self._joint_ids)
        # parse the body index
        body_ids, body_names = self._asset.find_bodies(self.cfg.body_name)
        if len(body_ids) != 1:
            raise ValueError(
                f"Expected one match for the body name: {self.cfg.body_name}. Found {len(body_ids)}: {body_names}."
            )
        # save only the first body index
        self._body_idx = body_ids[0]
        self._body_name = body_names[0]
        # check if articulation is fixed-base
        # if fixed-base then the jacobian for the base is not computed
        # this means that number of bodies is one less than the articulation's number of bodies
        if self._asset.is_fixed_base:
            self._jacobi_body_idx = self._body_idx - 1
            self._jacobi_joint_ids = self._joint_ids
        else:
            self._jacobi_body_idx = self._body_idx
            self._jacobi_joint_ids = [i + 6 for i in self._joint_ids]

        # log info for debugging
        omni.log.info(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f" {self._joint_names} [{self._joint_ids}]"
        )
        omni.log.info(
            f"Resolved body name for the action term {self.__class__.__name__}: {self._body_name} [{self._body_idx}]"
        )
        # Avoid indexing across all joints for efficiency
        if self._num_joints == self._asset.num_joints:
            self._joint_ids = slice(None)

        # create the differential IK controller
        self._ik_controller = DifferentialIKController(
            cfg=self.cfg.controller, num_envs=self.num_envs, device=self.device
        )

        # TODO confirm configuration steps yeild correct size output
        self.pos_dmp = DiscreteDMP(
            nRBF=cfg.dmp_cfg.num_weights,
            betaY=cfg.dmp_cfg.beta_y,
            dt=cfg.dmp_cfg.dt,
            cs=CS(
                ax = cfg.dmp_cfg.ax, 
                dt = 1 / cfg.decimation
            ),
            num_envs=self.num_envs,
            num_dims=3,
            device=self.device
        )

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros(
            (self.num_envs, cfg.decimation, self._ik_controller.action_dim),
            device=self.device
        )
        # save the scale as tensors
        self._scale = torch.zeros((self.num_envs, self.action_dim), device=self.device)
        self._scale[:] = torch.tensor(self.cfg.scale, device=self.device)

        # convert the fixed offsets to torch tensors of batched shape
        if self.cfg.body_offset is not None:
            self._offset_pos = torch.tensor(self.cfg.body_offset.pos, device=self.device).repeat(self.num_envs, 1)
            self._offset_rot = torch.tensor(self.cfg.body_offset.rot, device=self.device).repeat(self.num_envs, 1)
        else:
            self._offset_pos, self._offset_rot = None, None

        # parse clip
        if self.cfg.clip is not None:
            if isinstance(cfg.clip, dict):
                self._clip = torch.tensor([[-float("inf"), float("inf")]], device=self.device).repeat(
                    self.num_envs, self.action_dim, 1
                )
                index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.clip, self._joint_names)
                self._clip[:, index_list] = torch.tensor(value_list, device=self.device)
            else:
                raise ValueError(f"Unsupported clip type: {type(cfg.clip)}. Supported types are dict.")

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        # note this is the weights and the goal pose (pos, quat)
        return self.cfg.dmp_cfg.num_weights * 6 +7

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def jacobian_w(self) -> torch.Tensor:
        return self._asset.root_physx_view.get_jacobians()[:, self._jacobi_body_idx, :, self._jacobi_joint_ids]

    @property
    def jacobian_b(self) -> torch.Tensor:
        jacobian = self.jacobian_w
        base_rot = self._asset.data.root_quat_w
        base_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(base_rot))
        jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])
        return jacobian

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        pos_w_idx = self.pos_dmp.nRBF * self.pos_dmp.num_dims + 7
        
        self.pos_dmp.ws = (self.raw_actions[:,7:pos_w_idx] * self._scale[:,7:pos_w_idx]).view(self.num_envs, self.pos_dmp.nRBF, self.pos_dmp.num_dims)
        goal_pose = self.raw_actions[:,:7] * self._scale[:,:7]
        
        if self.cfg.clip is not None:
            self.pos_dmp.ws = torch.clamp(
                self.pos_dmp.ws, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )
            goal_pose = torch.clamp(
                goal_pose, min=self._clip[:,:,0], max=self._clip[:,:,1]
            )
        

        # obtain quantities from simulation
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        ee_vel_curr, ee_acc_curr = self._compute_frame_derivatives()

        # TODO calculate pose goals along trajectory
        _, raw_traj, _, _ = self.pos_dmp.rollout(
        #_, a, _, _ = self.pos_dmp.rollout(
            goal_pose[:,:3], 
            ee_pos_curr[:,:3], 
            ee_vel_curr[:,:3],
            ee_acc_curr[:,:3]
        )
        self._processed_actions[:,:,:3] = raw_traj[:,1:,:] # note we start at idx 1 because idx 0 is our current location
        self._processed_actions[:,:,3] = 1.0 # identity quat
        
        self.counter = 1
        # set command into controller
        self._ik_controller.set_command(self._processed_actions[:,0,:], ee_pos_curr, ee_quat_curr)

    def apply_actions(self):
        # obtain quantities from simulation
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        # compute the delta in joint-space

        # TODO
        # set current goal
        #self._ik_controller.set_command(self._processed_actions[:,self.counter,:], ee_pos_curr, ee_quat_curr)
        self.counter += 1

        if ee_quat_curr.norm() != 0:
            jacobian = self._compute_frame_jacobian()
            joint_pos_des = self._ik_controller.compute(ee_pos_curr, ee_quat_curr, jacobian, joint_pos)
        else:
            joint_pos_des = joint_pos.clone()
        # set the joint position command
        self._asset.set_joint_position_target(joint_pos_des, self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0

    """
    Helper functions.
    """

    def _compute_frame_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the pose of the target frame in the root frame.

        Returns:
            A tuple of the body's position and orientation in the root frame.
        """
        # obtain quantities from simulation
        ee_pos_w = self._asset.data.body_pos_w[:, self._body_idx]
        ee_quat_w = self._asset.data.body_quat_w[:, self._body_idx]
        root_pos_w = self._asset.data.root_pos_w
        root_quat_w = self._asset.data.root_quat_w
        # compute the pose of the body in the root frame
        ee_pose_b, ee_quat_b = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
        # account for the offset
        if self.cfg.body_offset is not None:
            ee_pose_b, ee_quat_b = math_utils.combine_frame_transforms(
                ee_pose_b, ee_quat_b, self._offset_pos, self._offset_rot
            )

        return ee_pose_b, ee_quat_b
    
    def _compute_frame_derivatives(self) -> tuple[torch.Tensor, torch.Tensor]:
        vel = torch.zeros((self.num_envs, 6), device=self.device)
        acc = torch.zeros((self.num_envs, 6), device=self.device)
        vel[:,:3] = self._asset.data.body_lin_vel_w[:,self._body_idx]
        vel[:,3:] = self._asset.data.body_ang_vel_w[:,self._body_idx]
        acc[:,:3] = self._asset.data.body_lin_acc_w[:,self._body_idx]
        acc[:,3:] = self._asset.data.body_ang_acc_w[:,self._body_idx]
        return vel, acc
        

    def _compute_frame_jacobian(self):
        """Computes the geometric Jacobian of the target frame in the root frame.

        This function accounts for the target frame offset and applies the necessary transformations to obtain
        the right Jacobian from the parent body Jacobian.
        """
        # read the parent jacobian
        jacobian = self.jacobian_b
        # account for the offset
        if self.cfg.body_offset is not None:
            # Modify the jacobian to account for the offset
            # -- translational part
            # v_link = v_ee + w_ee x r_link_ee = v_J_ee * q + w_J_ee * q x r_link_ee
            #        = (v_J_ee + w_J_ee x r_link_ee ) * q
            #        = (v_J_ee - r_link_ee_[x] @ w_J_ee) * q
            jacobian[:, 0:3, :] += torch.bmm(-math_utils.skew_symmetric_matrix(self._offset_pos), jacobian[:, 3:, :])
            # -- rotational part
            # w_link = R_link_ee @ w_ee
            jacobian[:, 3:, :] = torch.bmm(math_utils.matrix_from_quat(self._offset_rot), jacobian[:, 3:, :])

        return jacobian
