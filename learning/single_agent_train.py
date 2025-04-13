import argparse
import sys

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")

# exp
parser.add_argument("--task", type=str, default="TB2-FPiH-Franka-Rel_IK-v0", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=256, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=-1, help="Seed used for the environment")
parser.add_argument("--max_steps", type=int, default=10240000, help="RL Policy training iterations.")
parser.add_argument("--force_encoding", type=str, default=None, help="Which type of force encoding to use if force is included")
parser.add_argument("--num_agents", type=int, default=1, help="How many agents to train in parallel")
parser.add_argument("--learning_method", type=str, default="ppo", help="Which learning approach to use, currently ppo and sac are supported")
parser.add_argument("--dmp_obs", default=False, action="store_true", help="Should we use dmps for the observation space")
parser.add_argument("--init_eval", default=True, action="store_false", help="When added, we will not perform an eval before any training has happened")
parser.add_argument("--use_curriculum", default=False, action="store_true", help="Turns on z height curiculum")

# logging
parser.add_argument("--exp_name", type=str, default=None, help="What to name the experiment on WandB")
parser.add_argument("--exp_dir", type=str, default=None, help="Directory to store the experiment in")
parser.add_argument("--dump_yaml", action="store_true", default=False, help="Store config files in yaml format")
parser.add_argument("--dump_pickle", action="store_true", default=False, help="Store config files in pickle format")
parser.add_argument("--log_smoothness_metrics", action="store_true", default=False, help="Log the sum squared velocity, jerk and force metrics")
parser.add_argument("--no_vids", action="store_true", default=False, help="Set up sim environment to support cameras")

# wandb
parser.add_argument("--no_log_wandb", action="store_false", default=True, help="Disables the wandb logger")
parser.add_argument("--wandb_entity", type=str, default="hur", help="Name of wandb entity")
parser.add_argument("--wandb_project", type=str, default="Tester", help="Name of wandb project")
parser.add_argument("--wandb_api_key", type=str, default="-1", help="API key for WandB")
parser.add_argument('--wandb_tags', nargs='*', default=[], help="WandB Tags to be applied to this run")
parser.add_argument("--wandb_group", type=str, default=None, help="Group to organize wandb")


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
args_cli.wandb_tags = args_cli.wandb_tags[0].split(",")
#print("Tags:", args_cli.wandb_tags)
#assert 1 ==0
if not args_cli.no_vids:  
    args_cli.video = True
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args


# launch our threads before simulation app
from agents.mp_agent import MPAgent
import torch.multiprocessing as mp
mp_agent = None
if args_cli.num_agents > 1:
    n = args_cli.num_envs // args_cli.num_agents
    agents_scope = [[i * n, (i+1) * n] for i in range(args_cli.num_agents)]

    #mp.set_start_method("spawn")
    mp_agent = MPAgent(args_cli.num_agents, agents_scope=agents_scope )

# launch omniverse app
app_launcher = AppLauncher(args_cli)
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
from memory.adaptive_random_sample import AdaptiveRandomMemory
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.resources.preprocessors.torch import RunningStandardScaler

from learning.ext_sequential_trainer import ExtSequentialTrainer, EXT_SEQUENTIAL_TRAINER_DEFAULT_CONFIG
from wrappers.smoothness_obs_wrapper import SmoothnessObservationWrapper
from wrappers.close_gripper_action_wrapper import GripperCloseEnv
from models.default_mixin import Shared
from models.bro_model import BroAgent, BroActor, BroCritic
from models.SimBa import SimBaAgent
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

# seed for reproducibility
#set_seed(args_cli.seed)  # e.g. `set_seed(42)` for fixed seed
if args_cli.seed == -1:
    args_cli.seed = random.randint(0, 10000)
set_seed(args_cli.seed)
#agent_cfg_entry_point = "skrl_cfg_entry_point"
#agent_cfg_entry_point = f"BroNet_{args_cli.learning_method}_cfg_entry_point"
agent_cfg_entry_point = f"SimBaNet_ppo_cfg_entry_point"
#agent_cfg_entry_point = "rl_games_cfg_entry_point"
evaluating = False

@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, 
    agent_cfg: dict
):
    global evaluating
    global mp_agent

    """Train with skrl agent."""
    max_rollout_steps = agent_cfg['agent']['rollouts']
    print("Max Rollout steps:", max_rollout_steps)
    assert args_cli.num_envs % args_cli.num_agents == 0, f'Number of agents {args_cli.num_agents} does not even divide into number of envs {args_cli.num_envs}'
    env_per_agent = args_cli.num_envs // args_cli.num_agents
    
    args_cli.max_steps += max_rollout_steps * env_per_agent - (args_cli.max_steps % (max_rollout_steps * env_per_agent))

    # check inputs
    assert args_cli.max_steps % env_per_agent == 0, f'Iterations must be a multiple of num_envs: {env_per_agent}'
    assert args_cli.max_steps % max_rollout_steps == 0, f'Iterations must be multiple of max_rollout_steps {args_cli.max_steps % max_rollout_steps}'
    
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs 
    env_cfg.scene.replicate_physics = True
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    
    sim_dt = 1/50.0 
    policy_dt = 0.1#50*sim_dt
    dec =  int(policy_dt / sim_dt )
    episode_length_s = 5.0

    env_cfg.episode_length_s = episode_length_s
    env_cfg.sim.dt = sim_dt
    env_cfg.decimation = dec
    env_cfg.sim.render_interval = dec
    env_cfg.num_agents = args_cli.num_agents
    
    if args_cli.dmp_obs:
        env_cfg.scene.ee_imu.update_period = 0.0 # update every step
        env_cfg.scene.ee_imu.history_length = dec
        agent_cfg['agent']['logging_tags']['obs_type'] = "DMP"

    if "ActDMP" in args_cli.task:
        env_cfg.actions.arm_action.decimation = dec
        env_cfg.actions.arm_action.dmp_cfg.dt = sim_dt
        agent_cfg['agent']['logging_tags']['act_type'] = "DMP"

    env_cfg.use_curriculum = True
    if not agent_cfg['use_curriculum'] and not args_cli.use_curriculum:
        del env_cfg.curriculum.init_height_sampling
        env_cfg.use_curriculum = False

    if agent_cfg["break_force"] < 0:
        del env_cfg.rewards.broke_peg_failure

    if agent_cfg['seed'] >= 0:
        args_cli.seed = agent_cfg['seed']
        set_seed(args_cli.seed)
        agent_cfg['seed'] = args_cli.seed

    print("Seed:", agent_cfg['seed'], args_cli.seed)
    #print(env_cfg)
    # random sample some parameters
    agent_cfg['agent']['learning_rate_scheduler'] = KLAdaptiveLR
    """
    import numpy as np
    agent_cfg['agent']['param_space'] = {}
    agent_cfg['agent']['learning_rate_scheduler_kwargs']['kl_threshold'] = np.random.uniform(low=0.005, high=0.05)
    agent_cfg['agent']['param_space']['kl_threshold'] = agent_cfg['agent']['learning_rate_scheduler_kwargs']['kl_threshold']
    agent_cfg['agent']['entropy_loss_scale'] = np.random.uniform(low=0.0000005, high=0.00005)
    agent_cfg['agent']['param_space']['entropy_loss_scale'] = agent_cfg['agent']['entropy_loss_scale']
    agent_cfg['models']['act_init_std'] = np.random.uniform(low=0.15, high=0.35)
    agent_cfg['agent']['param_space']['act_init_std'] = agent_cfg['models']['act_init_std']
    #agent_cfg['agent']['mini_batches'] = int(np.random.choice([200, 100, 50]))
    agent_cfg['agent']['learning_rate'] = np.random.choice([0.001, 0.0005, 0.0001, 0.00005])
    agent_cfg['agent']['param_space']['learning_rate'] = agent_cfg['agent']['learning_rate']
    """

    #print("Decimation:", dec)
    agent_cfgs = [copy.deepcopy(agent_cfg) for _ in range(args_cli.num_agents)]
    # randomly sample a seed if seed = -1
    for agent_idx, a_cfg in enumerate(agent_cfgs):

        # specify directory for logging experiments
        if args_cli.exp_dir is None:
            log_root_path = os.path.join("logs", a_cfg["agent"]["experiment"]["directory"])
            if agent_idx > 0:
                log_root_path = os.path.join("logs", a_cfg["agent"]["experiment"]["directory"] + f"_{agent_idx}")
        else:
            log_root_path = os.path.join("logs", args_cli.exp_dir)
            if agent_idx > 0:
                log_root_path = os.path.join("logs", args_cli.exp_dir)

        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Logging experiment in directory: {log_root_path}")

        # specify directory for logging runs: {time-stamp}_{run_name}
        if args_cli.exp_name is None:
            if a_cfg["agent"]["experiment"]["experiment_name"] == "":
                log_dir = args_cli.task
            else:
                log_dir = f'{a_cfg["agent"]["experiment"]["experiment_name"]}'
        else:
            log_dir = f"{args_cli.exp_name}"

        log_dir += f'_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        
        # set directory into agent config
        a_cfg["agent"]["experiment"]["directory"] = log_root_path
        a_cfg["agent"]["experiment"]["experiment_name"] = log_dir
        # update log_dir
        log_dir = os.path.join(log_root_path, log_dir)
        #print("final log_dir=\n\t", log_dir)

        # agent configuration

        # dump the configuration into log-directory
        if args_cli.dump_yaml:
            dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
            dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), a_cfg)
        if args_cli.dump_pickle:
            dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
            dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), a_cfg)

        # determine video kwargs
        vid = not args_cli.no_vids
        print("Deciding to make vids")
        if vid:
            cfg = a_cfg['video_tracking']
            vid_interval = cfg['train_video_interval']
            vid_len = cfg['video_length']
            eval_vid = cfg['record_evals']
            train_vid = cfg['record_training']
        else:
            if agent_idx == 0:
                print("\n\nNo Videos will be recorded\n\n")
                delattr(env_cfg.observations.info, "img")
                eval_vid = False
                train_vid = False
                delattr(env_cfg.scene,"tiled_camera")
        
    # create env
    env = gym.make(
        args_cli.task, 
        cfg=env_cfg, 
        render_mode="rgb_array" if vid else None
    )
    
    if vid:
        # TODO: Setup dynamic config
        vid_fps = int(1.0 / (env.cfg.sim.dt * env.cfg.sim.render_interval ))

        print(f"\n*******Video Kwargs*******:\n\tvid:{vid}\n\tinterval:{vid_interval}")
        print(f"\teval:{eval_vid}\n\ttrain:{train_vid}\n\tlength:{vid_len}")
        print(f"\tFPS:{vid_fps}")
        print("***************************")
        
        def check_record(step):
            global evaluating
            if not evaluating:
                return step % vid_interval == 0
            return evaluating
        
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos"),
            "step_trigger": check_record, 
            "video_length": vid_len,
            "disable_logger": True,
            "fps": vid_fps
        }
        if eval_vid:
            os.makedirs(os.path.join(log_dir, "videos/evals"))
        if train_vid:
            os.makedirs(os.path.join(log_dir, "videos/training"))

        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        #env = ExtRecordVideo(env, **video_kwargs)
        env = InfoRecordVideo(env, **video_kwargs)
        vid_env = env
    else:
        vid_env = None
        

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

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(
        env, 
        ml_framework="torch"
    )  # same as: `wrap_env(env, wrapper="auto")    
    #env._reset_once = False
    env = GripperCloseEnv(env)
    
    device = env.device

    memory = RandomMemory( #AdaptiveRandomMemory( #
            memory_size=agent_cfg['agent']["rollouts"], 
            num_envs=env.num_envs // args_cli.num_agents, 
            device=device#,
            #replacement=True
        )
    # instantiate the agent's models (function approximators).
    # PPO requires 2 models, visit its documentation for more details
    # https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#models
    models = {}
    #models["policy"] = Shared(env.observation_space, env.action_space, device)
    
    
    # set wandb parameters
    for a_cfg in agent_cfgs:
        a_cfg['agent']['experiment']['wandb'] = args_cli.no_log_wandb
        wandb_kwargs = {
            "project":args_cli.wandb_project,
            "entity":args_cli.wandb_entity,
            "api_key":args_cli.wandb_api_key,
            "tags":args_cli.wandb_tags,
            "group":args_cli.wandb_group,
            "run_name":a_cfg["agent"]["experiment"]["experiment_name"]
        }

        a_cfg["agent"]["experiment"]["wandb_kwargs"] = wandb_kwargs
    
    agent_list = None
    if args_cli.learning_method == "ppo":
        print("Have a ppo actor")
        models['policy'] = SimBaAgent( #BroAgent(
            observation_space=env.observation_space, 
            action_space=env.action_space,
            device=device,
            act_init_std = agent_cfg['models']['act_init_std'],
            critic_output_init_mean = agent_cfg['models']['critic_output_init_mean']
            force_type = args_cli.force_encoding,
            critic_n = agent_cfg['models']['critic']['n'],
            critic_latent = agent_cfg['models']['critic']['latent_size'],
            actor_n = agent_cfg['models']['actor']['n'],
            actor_latent = agent_cfg['models']['actor']['latent_size']
        ) 
        models["value"] = models["policy"]  # same instance: shared model
        # create the agent
        #agent = WandbLoggerPPO(models=models,
        agent_list = [
            WandbLoggerPPO(
                models=copy.deepcopy(models),
                memory=copy.deepcopy(memory),
                cfg=agent_cfgs[i]['agent'],
                observation_space=env.observation_space,
                action_space=env.action_space,
                num_envs=args_cli.num_envs // args_cli.num_agents,
                device=device
            ) for i in range(args_cli.num_agents)
        ]
    elif args_cli.learning_method == "sac":
        print("Have a sac actor")
        models['policy'] = BroActor(
            observation_space=env.observation_space, 
            action_space=env.action_space,
            device=device,
            act_init_std = agent_cfg['models']['act_init_std'],
            force_type = args_cli.force_encoding,
            actor_n = agent_cfg['models']['actor']['n'],
            actor_latent = agent_cfg['models']['actor']['latent_size']
        ) 
        models["critic_1"] = BroCritic(
            observation_space=env.observation_space, 
            action_space=env.action_space,
            force_type = args_cli.force_encoding,
            device=device,
            critic_n = agent_cfg['models']['critic']['n'],
            critic_latent = agent_cfg['models']['critic']['latent_size']
        )
        models["critic_2"] = copy.deepcopy(models["critic_1"])
        models["target_critic_1"] = copy.deepcopy(models["critic_1"])
        models["target_critic_2"] = copy.deepcopy(models["critic_1"])
        # create the agent
        #agent = WandbLoggerPPO(models=models,
        agent_list = [
            WandbLoggerSAC(
                models=copy.deepcopy(models),
                memory=copy.deepcopy(memory),
                cfg=agent_cfgs[i]['agent'],
                observation_space=env.observation_space,
                action_space=env.action_space,
                num_envs=args_cli.num_envs // args_cli.num_agents,
                device=device
            ) for i in range(args_cli.num_agents)
        ]

    agents = None
    if args_cli.num_agents > 1:
        mp_agent.set_agents(agent_list)
        agents = mp_agent
    else:
        agents = agent_list[0]

    #TODO undo for vids later   
    if vid:
        vid_env.set_agent(agents)

    # configure and instantiate the RL trainer
    cfg_trainer = {
        "timesteps": args_cli.max_steps // (args_cli.num_envs * args_cli.num_agents), 
        "headless": True,
        "close_environment_at_exit": True
    }
    
    trainer = ExtSequentialTrainer(
        cfg = cfg_trainer,
        env = env,
        agents = agents
    )

    env.cfg.recording = True #vid # True
    # our actual learning loop
    ckpt_int = agent_cfg["agent"]["experiment"]["checkpoint_interval"]
    num_evals = max(1,args_cli.max_steps // (ckpt_int * env_per_agent))
    evaluating = True
    
    if eval_vid:   
       vid_env.set_video_name(f"evals/eval_0")

    if args_cli.init_eval:
        trainer.eval(0, vid_env)

    for i in range(num_evals):
        print(f"Beginning epoch {i+1}/{num_evals}")
        print("Training")
        evaluating = False
        if train_vid:     
            vid_env.set_video_name(f"training/train_STEP_NUM")
        trainer.train(ckpt_int, vid_env)
        
        evaluating = True
        if eval_vid:
            vid_env.set_video_name(f"evals/eval_{i+1}")

        print("Evaluating")
        trainer.eval(ckpt_int*(i+1), vid_env)



if __name__ == "__main__":
    # run the main function
    #import cProfile
    #import pstats

    #with cProfile.Profile() as pr:
    main()
    #    with open('tests/profile.txt', 'w') as f:
    #        pstats.Stats( pr, stream=f ).strip_dirs().sort_stats("cumtime").print_stats()
    #cProfile.run('main()', 'single_agent.profile')

    # close sim app
    simulation_app.close()