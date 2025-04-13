from __future__ import annotations

from dataclasses import MISSING

import torch
from dataclasses import dataclass
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.utils import configclass
from isaaclab.sensors import SensorBase
from isaaclab.sensors import SensorBaseCfg
from isaaclab.sensors.contact_sensor import ContactSensorCfg
#from isaacsim.core.articulations import ArticulationView
from isaacsim.core import RobotView
#if TYPE_CHECKING:
#    from sensors.force_torque_sensor.force_torque_cfg import ForceTorqueSensorCfg
from sensors.force_torque_sensor.force_torque_data import ForceTorqueSensorData

class ForceTorqueSensor(SensorBase):
    cfg: ForceTorqueSensorCfg
    def __init__(self, cfg: ForceTorqueSensorCfg):
        
        # ensure the imu data is only one big
        self._data = ForceTorqueSensorCfg()
        self.cfg = cfg
        self.fd = cfg.finite_diff
        self.history_length = max(2 if self.fd else 0, cfg.history_length) # we set history == 2 so we can do finite diff
        cfg.history_length = 0

        super().__init__(cfg)
        self._dt = None

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        return (
            f"Force Torque sensor @ '{self.cfg.prim_path}': \n"
            #f"\tview type         : {self._view.__class__}\n"
            f"\tupdate period (s) : {self.cfg.update_period}\n"
            f"\tnumber of sensors : {self._num_envs}\n"
            f"\thistory_length (sim steps) : {self.history_length}\n"
        )  
    
    @property
    def data(self) -> ForceTorqueSensorData:
        # update sensors if needed
        self._update_outdated_buffers()
        # return the data
        #print(self._data.net_forces_w.size())
        return self._data

    @property
    def num_instances(self) -> int:
        return self._num_envs

    def reset(self, env_ids: Sequence[int] | None = None):
        super().reset(env_ids)
        # resolve None
        if env_ids is None:
            env_ids = slice(None)
        # reset accumulative data buffers
        self._data.net_forces_w[env_ids,:] = 0.0
        if self.fd:
            self._data.net_forces_w_history[env_ids,:,:] = 0.0
            self._data.net_jerk_w_history[env_ids,:,:] = 0.0
            self._data.net_snap_w_history[env_ids,:,:] = 0.0

    def update(self, dt: float, force_recompute: bool = False):
        if self._dt == None:
            super().update(dt, force_recompute)
            self._dt = dt
            return
        # save timestamp
        self._dt = dt
        # execute updating
        #print(self._timestamp_last_update)
        super().update(dt, True)

    def _initialize_impl(self):
        # Initialize parent class
        super()._initialize_impl()

        #self._robot_av = ArticulationView(prim_paths_expr=self.cfg.prim_path) #, enable_dof_force_sensors=True)
        self._robot_av = RobotView(prim_paths_expr=self.cfg.prim_path)
        self._robot_av.initialize()
        # Create internal buffers
        self._initialize_buffers_impl()

    def _initialize_buffers_impl(self):
        """Create buffers for storing data."""
        # data buffers
        self._data.net_forces_w = torch.zeros(self._num_envs, 1, 6, device=self._device)
        self._data.net_forces_w_history = torch.zeros(self._num_envs, self.history_length, 6, device=self._device)
        if self.fd:
            self._data.net_jerk_w_history = torch.zeros(self._num_envs, self.history_length, 6, device=self._device)
            self._data.net_jerk_w = torch.zeros(self._num_envs, 1, 6, device=self._device)
            self._data.net_snap_w_history = torch.zeros(self._num_envs, self.history_length, 6, device=self._device)
            self._data.net_snap_w = torch.zeros(self._num_envs, 1, 6, device=self._device)

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        # roll the history forward, last in last out
        self._data.net_forces_w = self._robot_av.get_measured_joint_forces()[:, self.cfg.joint_idx, :]

        self._data.net_forces_w_history = torch.roll(self._data.net_forces_w_history, -1, 1)
        self._data.net_forces_w_history[:,-1,:] = self._data.net_forces_w

        if self.fd:
            self._data.net_jerk_w = (self._data.net_forces_w - self._data.net_forces_w_history[:,-2,:] ) / self._dt
            self._data.net_jerk_w_history = torch.roll(self._data.net_jerk_w_history, -1, 1)
            self._data.net_jerk_w_history[:,-1,:] = self._data.net_jerk_w
            
            self._data.net_jerk_w = (self._data.net_jerk_w - self._data.net_jerk_w_history[:,-2,:] ) / self._dt  
            self._data.net_snap_w_history = torch.roll(self._data.net_forces_w_history, -1, 1)
            self._data.net_snap_w_history[:,-1,:] = (self._data.net_forces_w - self._data.net_forces_w_history[:,-2, :]) / self._dt






@configclass
class ForceTorqueSensorCfg(SensorBaseCfg):
    class_type: type = ForceTorqueSensor
    joint_idx: int = 8