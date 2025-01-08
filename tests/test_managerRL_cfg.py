# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to run the RL environment for the cartpole balancing task."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on running the cartpole RL environment.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")
parser.add_argument("--config", type=str, default="FPiH", help="Which config to load and show")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

from omni.isaac.lab.envs import ManagerBasedRLEnv

from omni.isaac.lab_tasks.manager_based.classic.cartpole.cartpole_env_cfg import CartpoleEnvCfg
from envs.FPiH.FPiH_env_cfg import FragilePegInHoleEnvCfg
from envs.FPiH.config.franka.jnt_pos_env_cfg import FrankaFragilePegInHoleCfg

def main():
    """Main function."""
    # create environment configuration
    if args_cli.config == "FPiH":
        env_cfg = FrankaFragilePegInHoleCfg(num_envs=args_cli.num_envs, replicate_physics=False)
    else:
        env_cfg = CartpoleEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    # setup RL environment
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # simulate physics
    count = 0
    obs = None
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 100 == 0:
                #count = 0

                obs, _ = env.reset()
                #input("Reset (no Step)")
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            # sample random actions
            #print(obs['policy'][0,:7])
            poses = torch.tensor([
                0.0,
                -0.569,
                0.0,
                -2.810,
                0.0,
                3.037,
                0.741
            ])          
            #if count >= 100:
            #    input(count)

            joint_efforts = torch.zeros_like(env.action_manager.action)
            #for i in range(env.num_envs):
            #    joint_efforts[i, :] = torch.tensor([-0.0060,  0.5128,  0.0447,  2.7336, -0.1675, -2.9141, -0.6090])
            # step the environment
            obs, rew, terminated, truncated, info = env.step(joint_efforts)
            
            #print(obs['policy'].shape)
            #print("[Env 0]: Joint Pos: ", obs["policy"][0, :7])
            # update counter
            count += 1

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()