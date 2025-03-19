from dataclasses import MISSING

from omni.isaac.lab.controllers import DifferentialIKControllerCfg #, OperationalSpaceControllerCfg
from omni.isaac.lab.managers.action_manager import ActionTerm, ActionTermCfg
from omni.isaac.lab.utils import configclass

#from . import binary_joint_actions, joint_actions, joint_actions_to_limits, non_holonomic_actions, task_space_actions
import actions.DMP_space_actions as dmp_actions


@configclass
class DMPCfg:
    """The parameters for the Dynamic Motion Primative describing the trajectory"""
    num_weights: int = 10
    dims: int = 6
    dt: float = 1/200.0
    beta_y: float = 25.0/4.0
    ax: float = 5.0

@configclass
class DMPDiffIKActionCfg(ActionTermCfg):
    """Configuration for inverse differential kinematics action term.

    See :class:`DifferentialInverseKinematicsAction` for more details.
    """

    @configclass
    class OffsetCfg:
        """The offset pose from parent frame to child frame.

        On many robots, end-effector frames are fictitious frames that do not have a corresponding
        rigid body. In such cases, it is easier to define this transform w.r.t. their parent rigid body.
        For instance, for the Franka Emika arm, the end-effector is defined at an offset to the the
        "panda_hand" frame.
        """

        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Translation w.r.t. the parent frame. Defaults to (0.0, 0.0, 0.0)."""
        rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
        """Quaternion rotation ``(w, x, y, z)`` w.r.t. the parent frame. Defaults to (1.0, 0.0, 0.0, 0.0)."""

    class_type: type[ActionTerm] = dmp_actions.DMPDiffIKAction

    joint_names: list[str] = MISSING
    """List of joint names or regex expressions that the action will be mapped to."""
    body_name: str = MISSING
    """Name of the body or frame for which IK is performed."""
    body_offset: OffsetCfg | None = None
    """Offset of target frame w.r.t. to the body frame. Defaults to None, in which case no offset is applied."""
    scale: float | tuple[float, ...] = 1.0
    """Scale factor for the action. Defaults to 1.0."""
    controller: DifferentialIKControllerCfg = MISSING
    """The configuration for the differential IK controller."""
    dmp_cfg: DMPCfg = DMPCfg()
    """The configuration for the dynamic motion primative trajectories"""
    decimation: int = 20


