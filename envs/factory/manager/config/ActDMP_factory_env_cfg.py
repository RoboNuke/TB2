from isaaclab.utils import configclass
from envs.factory.manager import FactoryManagerEnvCfg

from isaaclab.envs import mdp
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg

from actions.DMP_space_actions_cfg import DMPDiffIKActionCfg, DMPCfg

@configclass
class DMPActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: DMPDiffIKActionCfg = DMPDiffIKActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_fingertip_centered",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale= 0.05,
        body_offset=DMPDiffIKActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        dmp_cfg = DMPCfg(
            num_weights = 10,
            dims = 6,
            dt = 1/200.0,
            beta_y = 25.0/4.0,
            ax = 5.0
        ),
        decimation = 20
        
    )
    
    #arm_action: mdp.JointPositionActionCfg = mdp.JointPositionActionCfg(
    #    asset_name="robot", joint_names=["panda_joint.*"], preserve_order=True
    #)
    gripper_action: mdp.BinaryJointPositionActionCfg = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )
    

@configclass
class FactoryManagerEnvActDMPCfg(FactoryManagerEnvCfg):

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.actions = DMPActionsCfg()
        self.actions.arm_action.decimation = self.decimation
        self.actions.arm_action.dmp_cfg.dt = self.sim.dt