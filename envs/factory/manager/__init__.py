import gymnasium as gym

from . import agents
#from .factory_env import FactoryEnv
from .factory_manager_env_cfg import FactoryManagerEnvCfg


gym.register(
    id="TB2-Factor-PiH-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FactoryManagerEnvCfg,
        "BroNet_cfg_entry_point": f"{agents.__name__}:BroNet_ppo_cfg.yaml"
    },
)