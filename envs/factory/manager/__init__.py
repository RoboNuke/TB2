import gymnasium as gym

from . import agents
#from .factory_env import FactoryEnv
from .factory_manager_env_cfg import FactoryManagerEnvCfg
from .config.ObsDMP_factory_env_cfg import FactoryManagerEnvObsDMPCfg
from .config.ActDMP_factory_env_cfg import FactoryManagerEnvActDMPCfg

gym.register(
    id="TB2-Factor-PiH-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FactoryManagerEnvCfg,
        "BroNet_ppo_cfg_entry_point": f"{agents.__name__}:BroNet_ppo_cfg.yaml",
        "BroNet_sac_cfg_entry_point": f"{agents.__name__}:BroNet_sac_cfg.yaml"
    },
)

gym.register(
    id="TB2-Factor-PiH-ObsDMP-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FactoryManagerEnvObsDMPCfg,
        "BroNet_ppo_cfg_entry_point": f"{agents.__name__}:BroNet_ppo_cfg.yaml",
        "BroNet_sac_cfg_entry_point": f"{agents.__name__}:BroNet_sac_cfg.yaml"
    },
)

gym.register(
    id="TB2-Factor-PiH-ActDMP-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FactoryManagerEnvActDMPCfg,
        "BroNet_ppo_cfg_entry_point": f"{agents.__name__}:BroNet_ppo_cfg.yaml",
        "BroNet_sac_cfg_entry_point": f"{agents.__name__}:BroNet_sac_cfg.yaml"
    },
)

