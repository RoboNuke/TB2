
from omni.isaac.lab.utils import configclass
from envs.factory.manager import FactoryManagerEnvCfg
from sensors.dense_pose_sensor.dense_pose_sensor import DummySensorCfg, DensePoseSensorCfg
from omni.isaac.lab.sensors import ImuCfg


from omni.isaac.lab.envs import mdp
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
import envs.factory.manager.mdp.observations as fac_mdp_obs
import envs.factory.manager.mdp.events as fac_mdp_events

@configclass
class DMPObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        
        ee_traj = ObsTerm(
            func = fac_mdp_obs.ee_traj
        )
        
        fingertip_pos_rel_fixed = ObsTerm(
            func = fac_mdp_obs.held_fixed_relative_pos
        )

        prev_action = ObsTerm(func=mdp.last_action)

        hole_pose = ObsTerm(
            func=fac_mdp_obs.fixed_asset_pose
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    
    @configclass
    class InfoCfg(ObsGroup):
        """Observations for information tracking"""

        dmg_force = ObsTerm(
            func=fac_mdp_obs.force_torque_sensor
        )

        joint_vel = ObsTerm(func=mdp.joint_vel_rel)

        joint_acc = ObsTerm(func=fac_mdp_obs.joint_acc_rel)

        img = ObsTerm(func=fac_mdp_obs.camera_image)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False


    # observation groups
    policy: PolicyCfg = PolicyCfg()
    info: InfoCfg = InfoCfg()

@configclass
class FactoryManagerEnvObsDMPCfg(FactoryManagerEnvCfg):

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # add imu to sensors
        self.scene.ee_imu = DensePoseSensorCfg(
            prim_path="/World/envs/env_.*/Robot/panda_hand",
            update_period = 0,
            history_length = 20,
            offset = ImuCfg.OffsetCfg(pos=[0.0, 0.0, 0.107])
        )
        # adjust observation
        self.observations = DMPObservationsCfg()
        # add init imu
        self.events.init_imu = EventTerm(
            func=fac_mdp_events.init_imu,
            mode="startup"
        )