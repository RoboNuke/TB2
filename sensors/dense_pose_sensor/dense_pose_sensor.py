from __future__ import annotations

from dataclasses import MISSING

import torch
from dataclasses import dataclass
from collections.abc import Sequence
from typing import TYPE_CHECKING

from omni.isaac.lab.utils import configclass
from omni.isaac.lab.sensors import SensorBase
from omni.isaac.lab.sensors import SensorBaseCfg
from omni.isaac.lab.sensors.imu import Imu, ImuData, ImuCfg

class DummySensor(Imu):
    old_dt = 0.0
    def __init__(self, cfg):
        print(f"\n\n\nIn the DUMMY sensor\n\n\n")
        super().__init__(cfg)
        print("Initialized Dummy Sensor")

    def update(self, dt: float, force_recompute: bool = False):
        # save timestamp
        print(f"\n\n\n{dt}\n\n\n")
        # execute updating
        super().update(dt, force_recompute)

@configclass
class DummySensorCfg(ImuCfg):
    class_type: type = DummySensor

class DensePoseSensor(SensorBase):
    cfg: ImuData
    def __init__(self, cfg: ImuData):
        
        # ensure the imu data is only one big
        self._data = DensePoseSensorData()
        
        self.history_length = cfg.history_length
        cfg.history_length = 0

        self.imu = Imu(cfg)
        super().__init__(cfg)
        self._dt = None

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        return (
            f"Dense Imu sensor @ '{self.cfg.prim_path}': \n"
            f"\tview type         : {self._view.__class__}\n"
            f"\tupdate period (s) : {self.cfg.update_period}\n"
            f"\tnumber of sensors : {self._view.count}\n"
            f"\thistory_length (sim steps) : {self.history_length}\n"
        )   

    @property
    def data(self) -> DensePoseSensorData:
        # update sensors if needed
        self._update_outdated_buffers()
        # return the data
        return self._data

    @property
    def num_instances(self) -> int:
        return self._view.count

    #def _update_outdated_buffers(self):


    def reset(self, env_ids: Sequence[int] | None = None):
        super().reset(env_ids)
        self.imu.reset(env_ids)
        # resolve None
        if env_ids is None:
            env_ids = slice(None)
        # reset accumulative data buffers
        self._data.quat_w[env_ids,:,:] = 0.0
        self._data.lin_vel_b[env_ids,:,:] = 0.0
        self._data.ang_vel_b[env_ids,:,:] = 0.0
        self._data.lin_acc_b[env_ids,:,:] = 0.0
        self._data.ang_acc_b[env_ids,:,:] = 0.0

    def update(self, dt: float, force_recompute: bool = False):
        if self._dt == None:
            super().update(dt, force_recompute)
            self.imu.update(dt, force_recompute)
            self._dt = dt
            return
        # save timestamp
        self._dt = dt
        # execute updating
        #print(self._timestamp_last_update)
        super().update(dt, True)
        self.imu.update(dt, True)

    def _initialize_impl(self):
        # Initialize parent class
        super()._initialize_impl()
        self.imu._initialize_impl()
        # Create internal buffers
        self._initialize_buffers_impl()

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        #print("update called")
        #super()._update_buffers_impl(env_ids)
        self.imu._update_buffers_impl(env_ids)
        # roll the history forward, last in last out
        self._data.pos_w = torch.roll(self._data.pos_w, -1, 1)
        self._data.pos_w[:,-1,:] = self.imu.data.pos_w
        self._data.quat_w = torch.roll(self._data.quat_w, -1, 1)
        self._data.quat_w[:,-1,:] = self.imu.data.quat_w
        self._data.lin_vel_b = torch.roll(self._data.lin_vel_b, -1, 1)  
        self._data.lin_vel_b[:,-1,:] = self.imu.data.lin_vel_b
        self._data.lin_acc_b = torch.roll(self._data.lin_acc_b, -1, 1)  
        self._data.lin_acc_b[:,-1,:] = self.imu.data.lin_acc_b
        self._data.ang_vel_b = torch.roll(self._data.ang_vel_b, -1, 1)  
        self._data.ang_vel_b[:,-1,:] = self.imu.data.ang_vel_b
        self._data.ang_acc_b = torch.roll(self._data.ang_acc_b, -1, 1)  
        self._data.ang_acc_b[:,-1,:] = self.imu.data.ang_acc_b

    def _initialize_buffers_impl(self):
        """Create buffers for storing data."""
        self.imu._initialize_buffers_impl()
        # data buffers
        self._data.pos_w = torch.zeros(self.imu._view.count, self.history_length, 3, device=self._device)
        self._data.quat_w = torch.zeros(self.imu._view.count, self.history_length, 4, device=self._device)
        self._data.quat_w[:, :, 0] = 1.0
        self._data.lin_vel_b = torch.zeros_like(self._data.pos_w)
        self._data.ang_vel_b = torch.zeros_like(self._data.pos_w)
        self._data.lin_acc_b = torch.zeros_like(self._data.pos_w)
        self._data.ang_acc_b = torch.zeros_like(self._data.pos_w)
        

    def _set_debug_vis_impl(self, debug_vis: bool):
        self.imu._set_debug_vis_impl(debug_vis)

    def _debug_vis_callback(self, event):
        self.imu._debug_vis_callback(event)

@configclass
class DensePoseSensorCfg(ImuCfg):
    class_type: type = DensePoseSensor

@dataclass
class DensePoseSensorData:
    """Data container for the dense pose reporting sensor."""

    pos_w: torch.Tensor = None
    """Position of the sensor origin in world frame.

    Shape is (N, T, 3), where N is the number of sensors, T is history length.
    """
    
    quat_w: torch.Tensor  = None
    """Orientation of the sensor origin in quaternion (w, x, y, z) in world frame.

    Shape is (N, T, 4), where N is the number of sensors. T is history length
    """

    lin_vel_b: torch.Tensor  = None
    """Linear velocity in world frame.

    Shape is (N, T, 3), where N is the number of sensors. T is history length
    """

    ang_vel_b: torch.Tensor  = None
    """Angular velocity in world frame.

    Shape is (N, T, 3), where N is the number of sensors. T is history length
    """

    lin_acc_b: torch.Tensor  = None
    """Linear acceleration in world frame.

    Shape is (N, T, 3), where N is the number of sensors. T is history length
    """

    ang_acc_b: torch.Tensor  = None
    """Angular acceleration in world frame.

    Shape is (N, T, 3), where N is the number of sensors. T is history length
    """
