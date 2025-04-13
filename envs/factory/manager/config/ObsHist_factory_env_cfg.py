
from isaaclab.utils import configclass
from envs.factory.manager import FactoryManagerEnvCfg
from sensors.dense_pose_sensor.dense_pose_sensor import DummySensorCfg, DensePoseSensorCfg
from isaaclab.sensors import ImuCfg


from isaaclab.envs import mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import EventTermCfg as EventTerm
import envs.factory.manager.mdp.observations as fac_mdp_obs
import envs.factory.manager.mdp.events as fac_mdp_events


@configclass
class FactoryManagerEnvObsHistCfg(FactoryManagerEnvCfg):

    def __post_init__(self):
        # post init of parent
        super().__post_init__()


        # add longer history to force-torque sensor
        self.scene.force_torque_sensor.history_length = self.cfg.decimation
        self.scene.ee_imu.history_length = self.cfg.decimation
        