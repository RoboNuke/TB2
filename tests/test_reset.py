from isaacsim import SimulationApp

# Parse any command-line arguments specific to the standalone application
import argparse
import argparse
import sys

from omni.isaac.lab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--test-arg", type=str, default="test", help="Test argument.")

AppLauncher.add_app_launcher_args(parser)

args, _ = parser.parse_known_args()
args.video= True
args.enable_cameras = True
# See DEFAULT_LAUNCHER_CONFIG for available configuration
# https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.isaac.kit/docs/index.html
launch_config = {"headless": False}
# Launch the Toolkit
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
from wrappers.video_recoder_wrapper import ExtRecordVideo
import os
import random
from datetime import datetime

import skrl
from packaging import version

from skrl.utils import set_seed
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.resources.preprocessors.torch import RunningStandardScaler

from learning.ext_sequential_trainer import ExtSequentialTrainer, EXT_SEQUENTIAL_TRAINER_DEFAULT_CONFIG
from wrappers.smoothness_obs_wrapper import SmoothnessObservationWrapper
from wrappers.close_gripper_action_wrapper import GripperCloseEnv
from models.default_mixin import Shared
from models.bro_model import BroAgent, BroActor, BroCritic
from wrappers.DMP_observation_wrapper import DMPObservationWrapper
from agents.agent_list import AgentList
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG

from omni.isaac.lab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml

import omni.isaac.lab_tasks  # noqa: F401
import envs.FPiH.config.franka
import envs.factory.direct
import envs.factory.manager
from omni.isaac.lab_tasks.utils.hydra import hydra_task_config
from omni.isaac.lab_tasks.utils.wrappers.skrl import SkrlVecEnvWrapper
from agents.wandb_logger_ppo_agent import WandbLoggerPPO
from agents.wandb_logger_sac_agent import WandbLoggerSAC

from tests.learning.toy_mdp import PrintActivity
from wrappers.info_video_recorder_wrapper import InfoRecordVideo
from agents.mp_agent import MPAgent
import torch.multiprocessing as mp
import copy
from skrl.resources.schedulers.torch import KLAdaptiveLR

import torch
seed = random.randint(0, 10000)
print(8817)
set_seed(8817)
agent_cfg_entry_point = f"BroNet_ppo_cfg_entry_point"
task = "TB2-Factor-PiH-v0"

@hydra_task_config(task, agent_cfg_entry_point)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, 
    agent_cfg: dict
):
    print("start hydra")
    env_cfg.scene.num_envs = 2
    env_cfg.scene.replicate_physics = True
    sim_dt = 1/50.0 
    policy_dt = 0.1#50*sim_dt
    dec =  int(policy_dt / sim_dt )
    episode_length_s = 0.2

    env_cfg.episode_length_s = episode_length_s
    env_cfg.sim.dt = sim_dt
    env_cfg.decimation = dec
    env_cfg.sim.render_interval = dec

    env = gym.make(
        task, 
        cfg=env_cfg, 
        render_mode="rgb_array" 
    )
    env = SkrlVecEnvWrapper(
        env, 
        ml_framework="torch"
    )  # same as: `wrap_env(env, wrapper="auto")    
    #env._reset_once = False
    env = GripperCloseEnv(env)
    env.reset()
    for i in range(1000):
        #print(f"step {i}")
        env.step(0.0 * torch.from_numpy(env.action_space.sample()).repeat(env.num_envs, 1))
        
    #for k in range(1000):
    #    env.step(0.0 * torch.from_numpy(env.action_space.sample()).repeat(env.num_envs, 1))
    #    cont = input("Press anything to continue...")
    #    print("reset")



if __name__ == "__main__":
    main()
    simulation_app.close()