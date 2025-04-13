import torch

from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import CONTACT_SENSOR_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.sensors import SensorBaseCfg, SensorBase
from isaaclab.sensors.contact_sensor import ContactSensorCfg
from sensors.force_torque_sensor.force_torque_sensor import ForceTorqueSensor

@configclass
class ForceTorqueSensorCfg(SensorBaseCfg):
    class_type: type = ForceTorqueSensor
    joint_idx: int = 8
    finite_diff: bool = False