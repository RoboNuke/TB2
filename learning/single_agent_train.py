import argparse
import sys

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
# exp
parser.add_argument("--task", type=str, default="TB2-FPiH-Franka-Rel_IK-v0", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=256, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=1, help="Seed used for the environment")
parser.add_argument("--max_steps", type=int, default=10240000, help="RL Policy training iterations.")
parser.add_argument("--force_encoding", type=str, default=None, help="Which type of force encoding to use if force is included")

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


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

if not args_cli.no_vids:  
    args_cli.video = True
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

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
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.resources.preprocessors.torch import RunningStandardScaler

from learning.ext_sequential_trainer import ExtSequentialTrainer, EXT_SEQUENTIAL_TRAINER_DEFAULT_CONFIG
from wrappers.smoothness_obs_wrapper import SmoothnessObservationWrapper
from wrappers.close_gripper_action_wrapper import GripperCloseEnv
from models.default_mixin import Shared
from models.bro_model import BroAgent
from wrappers.DMP_observation_wrapper import DMPObservationWrapper
from agents.agent_list import AgentList

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

from tests.learning.toy_mdp import PrintActivity
from wrappers.info_video_recorder_wrapper import InfoRecordVideo
from agents.mp_agent import MPAgent
import torch.multiprocessing as mp
# seed for reproducibility
set_seed(args_cli.seed)  # e.g. `set_seed(42)` for fixed seed

#agent_cfg_entry_point = "skrl_cfg_entry_point"
agent_cfg_entry_point = "BroNet_cfg_entry_point"
#agent_cfg_entry_point = "rl_games_cfg_entry_point"
evaluating = False

@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, 
    agent_cfg: dict
):
    global evaluating

    mp.set_start_method("spawn")
    """Train with skrl agent."""
    max_rollout_steps = agent_cfg['agent']['rollouts']

    # check inputs
    assert args_cli.max_steps % args_cli.num_envs == 0, f'Iterations must be a multiple of num_envs: {args_cli.max_steps % args_cli.num_envs}'
    assert args_cli.max_steps % max_rollout_steps == 0, f'Iterations must be multiple of max_rollout_steps {args_cli.max_steps % max_rollout_steps}'
    
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs 
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
    
    if "ObsDMP" in args_cli.task:
        env_cfg.scene.ee_imu.update_period = 0.0 # update every step
        env_cfg.scene.ee_imu.history_length = dec

    if "ActDMP" in args_cli.task:
        env_cfg.actions.arm_action.decimation = dec
        env_cfg.actions.arm_action.dmp_cfg.dt = sim_dt

    #print("Decimation:", dec)

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    agent_cfg["seed"] = args_cli.seed
    env_cfg.seed = agent_cfg["seed"]

    # specify directory for logging experiments
    if args_cli.exp_dir is None:
        log_root_path = os.path.join("logs", agent_cfg["agent"]["experiment"]["directory"])
    else:
        log_root_path = os.path.join("logs", args_cli.exp_dir)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    # specify directory for logging runs: {time-stamp}_{run_name}
    if args_cli.exp_name is None:
        if agent_cfg["agent"]["experiment"]["experiment_name"] == "":
            log_dir = args_cli.task
        else:
            log_dir = f'{agent_cfg["agent"]["experiment"]["experiment_name"]}'
    else:
        log_dir = f"{args_cli.exp_name}"

    log_dir += f'_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    
    # set directory into agent config
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    # update log_dir
    log_dir = os.path.join(log_root_path, log_dir)
    #print("final log_dir=\n\t", log_dir)

    # agent configuration

    # dump the configuration into log-directory
    if args_cli.dump_yaml:
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    if args_cli.dump_pickle:
        dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
        dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # determine video kwargs
    vid = not args_cli.no_vids
    if vid:
        cfg = agent_cfg['video_tracking']
        vid_interval = cfg['train_video_interval']
        vid_len = cfg['video_length']
        eval_vid = cfg['record_evals']
        train_vid = cfg['record_training']
    else:
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
        
    #if args_cli.dmp_obs:
    if "ObsDMP" in args_cli.task:
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

    memory = RandomMemory(
            memory_size=agent_cfg['agent']["rollouts"], 
            num_envs=env.num_envs // 2, 
            device=device
        )
    # instantiate the agent's models (function approximators).
    # PPO requires 2 models, visit its documentation for more details
    # https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#models
    models = {}
    #models["policy"] = Shared(env.observation_space, env.action_space, device)
    
    
    models['policy'] = BroAgent(
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
    models["value"] = models["policy"]  # same instance: shared model
    
    # set wandb parameters
    agent_cfg['agent']['experiment']['wandb'] = args_cli.no_log_wandb
    wandb_kwargs = {
        "project":args_cli.wandb_project,
        "entity":args_cli.wandb_entity,
        "api_key":args_cli.wandb_api_key,
        "tags":args_cli.wandb_tags,
        "run_name":agent_cfg["agent"]["experiment"]["experiment_name"]
    }

    agent_cfg["agent"]["experiment"]["wandb_kwargs"] = wandb_kwargs
    # create the agent
    #agent = WandbLoggerPPO(models=models,
    import copy
    agent_list = [
        WandbLoggerPPO(
            models=copy.deepcopy(models),
            memory=copy.deepcopy(memory),
            cfg=agent_cfg['agent'],
            observation_space=env.observation_space,
            action_space=env.action_space,
            num_envs=2,
            device=device
        ) for _ in range(2)
    ]
    
    agent = MPAgent(agents=agent_list, agents_scope=[[0,2],[2,4]] )
    # TODO make scope and num agents a var
    if vid:
        vid_env.set_agent(AgentList(agent_list, agents_scope=[2,2]))

    # configure and instantiate the RL trainer
    cfg_trainer = {
        "timesteps": args_cli.max_steps // args_cli.num_envs, 
        "headless": True,
        "close_environment_at_exit": True
    }

    # TODO make scope a var
    print("Action Space:", env.action_space)
    print("Obs Space:", env.observation_space)
    trainer = ExtSequentialTrainer(
        cfg = cfg_trainer,
        env = env,
        agents = agent
    )

    env.recording = vid # True
    # our actual learning loop
    ckpt_int = agent_cfg["agent"]["experiment"]["checkpoint_interval"]
    num_evals = max(1,args_cli.max_steps // (ckpt_int * args_cli.num_envs))
    evaluating = True
    if eval_vid:   
        vid_env.set_video_name(f"evals/eval_0")
    
    #trainer.eval(0, vid_env)

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