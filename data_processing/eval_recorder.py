import argparse
import sys

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")

# exp
parser.add_argument("--task", type=str, default="TB2-Factor-PiH-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=0, help="Seed used for the environment")
parser.add_argument("--force_encoding", type=str, default=None, help="Which type of force encoding to use if force is included")
parser.add_argument("--learning_method", type=str, default="ppo", help="Which learning approach to use, currently ppo and sac are supported")

# logging
parser.add_argument("--exp_name", type=str, default="rel_obs_param_tests_2025-04-07_01-08-20", help="What to name the experiment on WandB")
parser.add_argument("--exp_dir", type=str, default="/home/hunter/TB2/logs/Tests", help="Directory to store the experiment in")

# wandb
parser.add_argument("--no_log_wandb", action="store_false", default=True, help="Disables the wandb logger")
parser.add_argument("--wandb_entity", type=str, default="hur", help="Name of wandb entity")
parser.add_argument("--wandb_project", type=str, default="Tester", help="Name of wandb project")
parser.add_argument("--wandb_api_key", type=str, default="-1", help="API key for WandB")


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

#args_cli.video = True
args_cli.enable_cameras = True


# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import torch
import torchvision.transforms.functional as F
from PIL import Image
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

from wrappers.smoothness_obs_wrapper import SmoothnessObservationWrapper
from wrappers.close_gripper_action_wrapper import GripperCloseEnv
from models.default_mixin import Shared
from models.bro_model import BroAgent
from wrappers.DMP_observation_wrapper import DMPObservationWrapper

from omni.isaac.lab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)

import omni.isaac.lab_tasks  # noqa: F401
import envs.FPiH.config.franka
import envs.factory.direct
import envs.factory.manager
from omni.isaac.lab_tasks.utils.hydra import hydra_task_config
from omni.isaac.lab_tasks.utils.wrappers.skrl import SkrlVecEnvWrapper
from agents.wandb_logger_ppo_agent import WandbLoggerPPO
import tqdm

import matplotlib.pyplot as plt
def save_tensor_as_gif(tensor_list, filename, duration=100, loop=0):
    """
    Saves a list of PyTorch tensors as a GIF image.

    Args:
        tensor_list (list of torch.Tensor): List of tensors to be saved as frames.
        filename (str): Output filename for the GIF.
        duration (int, optional): Duration of each frame in milliseconds. Defaults to 100.
        loop (int, optional): Number of times the GIF should loop. 0 means infinite loop. Defaults to 0.
    """
    tensor_list = tensor_list.permute(0, 3, 1, 2)
    tensor_list = tensor_list
    images = []
    print(torch.max(tensor_list), torch.min(tensor_list))
    for i in range(50):#tensor_list:
        # Ensure the tensor is in CPU and convert it to a PIL Image
        tensor = tensor_list[i,:,:,:] 
        img = F.to_pil_image(tensor.to("cpu"))
        images.append(img)
        #plt.plot(img)
        #plt.show()

    # Save the list of PIL Images as a GIF
    images[0].save(filename, save_all=True, append_images=images[1:], duration=duration, loop=loop)

agent_cfg_entry_point = f"BroNet_{args_cli.learning_method}_cfg_entry_point"

class Img2InfoWrapperclass(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def step(self, action):
        """Steps through the environment using action, recording observations if :attr:`self.recording`."""
        (
            observations,
            rewards,
            terminateds,
            truncateds,
            infos,
        ) = self.env.step(action)
        infos['img'] = observations['info']['img']
        return observations, rewards, terminateds, truncateds, infos 
    def reset(self, **kwargs):
        """Reset the environment using kwargs and then starts recording if video enabled."""
        observations, info = super().reset(**kwargs)
        info['img'] = observations['info']['img']
        return observations, info

@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, 
    agent_cfg: dict
):
    import wandb
    # open wandb run (set to append)
    
    api = wandb.Api()

    name = args_cli.wandb_entity + "/" + args_cli.wandb_project
    des_exp_names = [
        "Std_Obs_2025-04-07_01-26-48", 
        #"Std_Obs_2025-04-07_14-42-40", 
        #"Std_Obs_2025-04-07_14-58-44",
        #"Std_Obs_2025-04-07_21-19-47",
        #"Std_Obs_2025-04-07_21-20-18"
        "Std_Obs_2025-04-07_01-54-18"
    ]
    tests_done = {name:[False, False, False, False] for name in des_exp_names}
    des_filepath = "/nfs/hpc/share/brownhun/TB2/logs/"
    des_tags = ['obs_param_test']
    exc_tags = ['failed_reset']
    
    runs = api.runs(name)
    run_ids_to_run = []
    for run in runs:
        exp_dir = run.config['experiment']['directory']
        exp_name = run.config['experiment']['experiment_name']
        tags = run.tags
        print(exp_name)
        #print(exp_name, exp_name in des_exp_names, des_filepath in exp_dir, exp_dir)
        for tag in des_tags:
            if not tag in tags:
                continue
        #print("\tincludes all tags")
        for tag in exc_tags:
            if tag in tags:
                continue
        #print("\tdoesn't have bad tags")

        if des_filepath in exp_dir and exp_name in des_exp_names:
            #print("\tHas dir and name")
            ll = 0 if exp_dir[-1] == "s" else int(exp_dir[-1])
            #print("\tll:", ll)
            if not tests_done[exp_name][ll]:
                tests_done[exp_name][ll] = True
                print(exp_name, run.id, ll)
                run_ids_to_run.append(run.id)


        #configs = {k: v for k, v in run.config.items() if not k.startswith("_")}
        #print(exp_name, exp_dir == args_cli.exp_dir, exp_name==args_cli.exp_name)
        #if exp_dir == args_cli.exp_dir and exp_name == args_cli.exp_name:
        # #   run_id = run.id
        #    print(run_id)

    #assert 1 == 0
    
    # create env
    max_rollout_steps = agent_cfg['agent']['rollouts']
    env_cfg.scene.num_envs = 8
    env_cfg.scene.replicate_physics = True
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    
    sim_dt = 1/200.0 
    policy_dt = 0.1#50*sim_dt
    dec =  int(policy_dt / sim_dt )
    episode_length_s = 5.0

    env_cfg.episode_length_s = episode_length_s
    env_cfg.sim.dt = sim_dt
    env_cfg.decimation = dec
    env_cfg.sim.render_interval = dec
    agent_cfg['agent']['learning_rate_scheduler'] = None
    """
    if args_cli.dmp_obs:
        env_cfg.scene.ee_imu.update_period = 0.0 # update every step
        env_cfg.scene.ee_imu.history_length = dec
        agent_cfg['agent']['logging_tags']['obs_type'] = "DMP"

    if "ActDMP" in args_cli.task:
        env_cfg.actions.arm_action.decimation = dec
        env_cfg.actions.arm_action.dmp_cfg.dt = sim_dt
        agent_cfg['agent']['logging_tags']['act_type'] = "DMP"
    """
    #print("[INFO]: Config complete")
    # create env
    env = gym.make(
        args_cli.task, 
        cfg=env_cfg, 
        render_mode="rgb_array"
    )
    """
    if args_cli.log_smoothness_metrics:
        print("\n\n[INFO] Recording Smoothness Metrics in info.\n\n")
        env = SmoothnessObservationWrapper(env)
        
    if args_cli.dmp_obs:
        print("\n\n[INFO] Using DMP observation wrapper.\n\n")
        env = DMPObservationWrapper(
            env=env,
            num_weights=10,
            fit_force_data=False,
            sim_dt=env_cfg.sim.dt,
            update_dt=policy_dt
        )
    
    if "ActDMP" in args_cli.task:
        print("Action DMPs currently unsupported")

    if "VIC" in args_cli.task:
        print("VIC is currently not supported")
    """
    env = Img2InfoWrapperclass(env)
    env = SkrlVecEnvWrapper(
        env, 
        ml_framework="torch"
    )  # same as: `wrap_env(env, wrapper="auto")    
    #env._reset_once = False
    env = GripperCloseEnv(env)
    
    env.cfg.recording = True
    device = env.device
    
    model = {}
    model['policy'] = BroAgent(
            observation_space=env.observation_space, 
            action_space=env.action_space,
            device=device,
            act_init_std = agent_cfg['models']['act_init_std'],
            force_type = args_cli.force_encoding,
            critic_n = agent_cfg['models']['critic']['n'],
            critic_latent = agent_cfg['models']['critic']['latent_size'],
            actor_n = agent_cfg['models']['actor']['n'],
            actor_latent = agent_cfg['models']['actor']['latent_size']
        ) 
    model["value"] = model["policy"]  # same instance: shared model

    agent_cfg['agent']['experiment']['wandb'] = False
    agent = WandbLoggerPPO(
        models=model,
        memory=None,
        cfg=agent_cfg['agent'],
        observation_space=env.observation_space,
        action_space=env.action_space,
        num_envs=env.num_envs,
        device=device
    )
    
    
    images = torch.zeros((max_rollout_steps, 2*180, 4*240, 3), device = env.device)
    for idx, run_id in enumerate(run_ids_to_run):
        print(f"Starting run:{run_id}; \t{idx}/{len(run_ids_to_run)}")
        
        run = wandb.init(
            entity=args_cli.wandb_entity, 
            project=args_cli.wandb_project, 
            id=run_id, 
            resume="must"
        )
        
        # get list of checkpoints:
        fp = args_cli.exp_dir + "/" + args_cli.exp_name + "/checkpoints"
        #print("File Path: ", fp)
        checkpoint_files = [f for f in os.listdir(fp) if os.path.isfile(os.path.join(fp, f))]
        #print("Checkpoint_files:", checkpoint_files)
        # for each checkpoint:

        with torch.no_grad():
            for ckpt_fp_idx in tqdm.tqdm(range(len(checkpoint_files)), file=sys.stdout):
                ckpt_fp = checkpoint_files[ckpt_fp_idx]
                #print("Current Checkpoint File:", ckpt_fp)
                #   load agent
                agent.load(fp + "/" + ckpt_fp)
                # reset env
                states, infos = env.reset()
                
                alive_mask = torch.ones(size=(states.shape[0], 1), device=states.device, dtype=bool)
                for i in tqdm.tqdm(range(max_rollout_steps), file=sys.stdout):
                        # get action
                        actions = agent.act(
                            states, 
                            timestep=1000, 
                            timesteps=1000
                        )[-1]['mean_actions']
                        
                        actions[~alive_mask[:,0],:] *= 0.0
                        
                        # take action
                        next_states, _, terminated, truncated, _ = env.step(actions)
                
                        for k in range(env.num_envs):
                            y = k // 4
                            x = k % 4
                            images[i, y*180:(y+1)*180, x*240:(x+1)*240, :] = infos['img'][k,:,:,:]

                        mask_update = ~torch.logical_or(terminated, truncated)
                        
                        alive_mask *= mask_update
                        # get image + value est.
                        # reset environments
                        if env.num_envs > 1:
                            states = next_states
                        else:
                            if terminated.any() or truncated.any():
                                with torch.no_grad():
                                    states, infos = env.reset()
                            else:
                                states = next_states
                    # draw eval est + actions on image
                # make imgs into gif
                img_path = f'{args_cli.exp_dir}/{args_cli.exp_name}/{ckpt_fp[:-3]}.gif'
                save_tensor_as_gif(images, img_path)
                #print("Saved to:", img_path)

                #assert 1 ==0
                # add gif to wandb 
                wandb.log({
                    "eval_video":wandb.Video(
                        data_or_path=img_path,
                        caption=ckpt_fp[:-3],
                        #fps=10,
                        format='gif'
                    ),
                    "video_step": int(ckpt_fp[6:-3])
                })
        wandb.finish()

if __name__ == "__main__":
    main()
    simulation_app.close()