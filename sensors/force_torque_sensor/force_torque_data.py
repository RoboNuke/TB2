
from __future__ import annotations

import torch
from dataclasses import dataclass

from omni.isaac.lab.sensors.contact_sensor import ContactSensorData


@dataclass
class ForceTorqueSensorData(ContactSensorData):
    net_forces_w: torch.Tensor | None = None
    net_jerk_w: torch.Tensor | None = None
    net_snap_w: torch.Tensor | None = None
    """The net normal contact forces in world frame.

    Shape is (N, B, 6), where N is the number of sensors and B is the number of bodies in each sensor.

    Note:
        This quantity is the sum of the normal contact forces acting on the sensor bodies. It must not be confused
        with the total contact forces acting on the sensor bodies (which also includes the tangential forces).
    """

    net_forces_w_history: torch.Tensor | None = None
    net_jerk_w_history: torch.Tensor | None = None
    net_snap_w_history: torch.Tensor | None = None
    """The net normal contact forces in world frame.

    Shape is (N, T, B, 6), where N is the number of sensors, T is the configured history length
    and B is the number of bodies in each sensor.

    In the history dimension, the first index is the most recent and the last index is the oldest.

    Note:
        This quantity is the sum of the normal contact forces acting on the sensor bodies. It must not be confused
        with the total contact forces acting on the sensor bodies (which also includes the tangential forces).
    """