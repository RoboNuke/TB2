import torch

from omni.isaac.lab.markers import VisualizationMarkersCfg
from omni.isaac.lab.markers.config import CONTACT_SENSOR_MARKER_CFG
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.sensors import SensorBaseCfg, SensorBase
from omni.isaac.lab.sensors.contact_sensor import ContactSensorCfg
from sensors.force_torque_sensor.force_torque_sensor import ForceTorqueSensor

@configclass
class ForceTorqueSensorCfg(SensorBaseCfg):
    class_type: type = ForceTorqueSensor
    joint_idx: int = 8