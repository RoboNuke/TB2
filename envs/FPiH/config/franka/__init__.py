import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


##
# Joint Position Control
##

gym.register(
    id="TB2-FPiH-Franka-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.jnt_pos_env_cfg:FrankaFragilePegInHoleCfg",
        #"rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        #"rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaReachPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "BroNet_cfg_entry_point": f"{agents.__name__}:BroNet_ppo_cfg.yaml"
    },
)

gym.register(
    id="TB2-FPiH-Franka-Rel_IK-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rel_ik_env_cfg:FrankaFragilePegInHoleCfg",
        #"rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        #"rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaReachPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "BroNet_cfg_entry_point": f"{agents.__name__}:BroNet_ppo_cfg.yaml"
    },
)