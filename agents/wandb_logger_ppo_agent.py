from typing import Any, Mapping, Optional, Tuple, Union
import copy
import os

import gym
import gymnasium
import skrl
import torch
import numpy as np
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl import config, logger

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from data_processing.data_manager import DataManager

import wandb

import collections
class WandbLoggerPPO(PPO):
    def __init__(
            self,
            models: Mapping[str, Model],
            memory: Optional[Union[Memory, Tuple[Memory]]] = None,
            observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
            action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
            device: Optional[Union[str, torch.device]] = None,
            cfg: Optional[dict] = None
    ) -> None:
        super().__init__(models, memory, observation_space, action_space, device, cfg)
        self.global_step = 0
        self.num_envs = memory.num_envs

        self._track_rewards = collections.deque(maxlen=1000)
        self._track_timesteps = collections.deque(maxlen=1000)

    def init(self, trainer_cfg: Optional[Mapping[str, Any]] = None) -> None:
        """Initialize the agent

        This method should be called before the agent is used.
        It will initialize the TensorBoard writer (and optionally Weights & Biases) and create the checkpoints directory

        :param trainer_cfg: Trainer configuration
        :type trainer_cfg: dict, optional
        """

        #super().init(trainer_cfg)
        #return
        trainer_cfg = trainer_cfg if trainer_cfg is not None else {}

        # update agent configuration to avoid duplicated logging/checking in distributed runs
        if config.torch.is_distributed and config.torch.rank:
            self.write_interval = 0
            self.checkpoint_interval = 0
            # TODO: disable wandb
            

        # main entry to log data for consumption and visualization by TensorBoard
        if self.write_interval == "auto":
            self.write_interval = int(trainer_cfg.get("timesteps", 0) / 100)
        #if self.write_interval > 0:
        #    self.writer = SummaryWriter(log_dir=self.experiment_dir)

        if self.checkpoint_interval == "auto":
            self.checkpoint_interval = int(trainer_cfg.get("timesteps", 0) / 10)
        if self.checkpoint_interval > 0:
            os.makedirs(os.path.join(self.experiment_dir, "checkpoints"), exist_ok=True)

        self.set_mode("eval")

        # create tensors in memory
        if self.memory is not None:
            self.memory.create_tensor(name="states", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
            self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="log_prob", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="values", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="returns", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="advantages", size=1, dtype=torch.float32)

            # tensors sampled during training
            self._tensors_names = ["states", "actions", "log_prob", "values", "returns", "advantages"]

        # create temporary variables needed for storage and computation
        self._current_log_prob = None
        self._current_next_states = None



        # setup Weights & Biases
        
        if self.cfg.get("experiment", {}).get("wandb", False):
            # save experiment configuration
            try:
                models_cfg = {k: v.net._modules for (k, v) in self.models.items()}
            except AttributeError:
                models_cfg = {k: v._modules for (k, v) in self.models.items()}
            wandb_config={**self.cfg, **trainer_cfg, **models_cfg}
            # set default values
            wandb_kwargs = copy.deepcopy(self.cfg.get("experiment", {}).get("wandb_kwargs", {}))
            wandb_kwargs.setdefault("name", os.path.split(self.experiment_dir)[-1])
            wandb_kwargs.setdefault("config", {})
            wandb_kwargs["config"].update(wandb_config)
            # init Weights & Biases
            
            
            if wandb_kwargs['api_key'] == "-1":
                self.data_manager = DataManager(
                    project=wandb_kwargs['project'],
                    entity=wandb_kwargs['entity']
                )
            else:
                self.data_manager = DataManager(
                    project=wandb_kwargs['project'],
                    entity=wandb_kwargs['entity'],
                    api_key=wandb_kwargs['api_key']
                )
            
            self.data_manager.init_new_run(
                run_name=wandb_kwargs['run_name'],
                config=wandb_config,
                tags=wandb_kwargs['tags']
            )

    def track_video_path(self, tag: str, value: str, timestep)-> None:
        self.tracking_data[tag + "_video"].append(value)
        self.tracking_data[tag + "_video"].append(timestep * self.num_envs)
        #self.data_manager.add_mp4(value, step= timestep * self.num_envs, cap=tag, fps=30)

    def track_hist(self, tag: str, value: torch.Tensor, timestep: int) -> None:
        self.data_manager.add_histogram(tag, value, timestep * self.num_envs)

    def track_data(self, tag: str, value: float) -> None:
        """Track data to TensorBoard

        Currently only scalar data are supported

        :param tag: Data identifier (e.g. 'Loss / policy loss')
        :type tag: str
        :param value: Value to track
        :type value: float
        """
        self.tracking_data[tag].append(value)

    def write_tracking_data(self, timestep: int, timesteps: int, eval=False) -> None:
        """Write tracking data to TensorBoard

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        prefix = "Training"
        if eval:
            prefix = "Eval"

        # handle cumulative rewards
        if len(self._track_rewards):
            track_rewards = np.array(self._track_rewards)
            track_timesteps = np.array(self._track_timesteps)
            
            self.tracking_data[prefix + " Reward / Return (max)"].append(np.max(track_rewards))
            self.tracking_data[prefix + " Reward / Return (min)"].append(np.min(track_rewards))
            self.tracking_data[prefix + " Reward / Return (mean)"].append(np.mean(track_rewards))

            self.tracking_data[prefix + " Episode / Total timesteps (max)"].append(np.max(track_timesteps))
            self.tracking_data[prefix + " Episode / Total timesteps (min)"].append(np.min(track_timesteps))
            self.tracking_data[prefix + " Episode / Total timesteps (mean)"].append(np.mean(track_timesteps))

        keep = {}
        for k, v in self.tracking_data.items():
            if k.endswith("_video"):
                try:
                    self.data_manager.add_mp4(
                        v[0], # path
                        k[:-6], #tag
                        timestep * self.num_envs, # step
                        f"At timestep {v[1]}" # caption
                    )
                except FileNotFoundError:
                    keep[k] = v
            elif k.endswith("(min)"):
                self.data_manager.add_scalar({k:np.min(v)}, timestep * self.num_envs)
            elif k.endswith("(max)"):
                self.data_manager.add_scalar({k:np.max(v)}, timestep * self.num_envs)
            else:
                self.data_manager.add_scalar({k:np.mean(v)}, timestep * self.num_envs)

            if k in self.m4_returns:
                self.m4_returns[k] *= 0
            
        
        # handle counting stats

        for k,v in self.count_stats.items():
            idx = k.index("/") + 1 
            my_k = prefix + k
            my_k = my_k.replace('Episode_Termination/', ' Termination /')
            #self.track_data(my_k, v.item())
            self.data_manager.add_scalar({my_k:v.item()}, timestep * self.num_envs)
            self.count_stats[k] *= 0

        # reset data containers for next iteration
        self._track_rewards.clear()
        self._track_timesteps.clear()
        self.tracking_data.clear()
        

        for k, v in keep.items():
            self.tracking_data[k] = v

    def post_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called after the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        self._rollout += 1
        if not self._rollout % self._rollouts and timestep >= self._learning_starts:
            self.set_mode("train")
            self._update(timestep, timesteps)
            self.set_mode("eval")

        timestep += 1
        self.global_step+= self.num_envs

        # update best models and write checkpoints
        if timestep > 1 and self.checkpoint_interval > 0 and not timestep % self.checkpoint_interval:
            # update best models
            reward = np.mean(self.tracking_data.get("Reward / Total reward (mean)", -2 ** 31))
            if reward > self.checkpoint_best_modules["reward"]:
                self.checkpoint_best_modules["timestep"] = timestep
                self.checkpoint_best_modules["reward"] = reward
                self.checkpoint_best_modules["saved"] = False
                self.checkpoint_best_modules["modules"] = {k: copy.deepcopy(self._get_internal_value(v)) for k, v in self.checkpoint_modules.items()}
            # write checkpoints
            self.write_checkpoint(timestep, timesteps)

        # write wandb
        if timestep > 1 and self.write_interval > 0 and not timestep % self.write_interval:
            self.write_tracking_data(timestep, timesteps)

    def record_transition(self,
                        states: torch.Tensor,
                        actions: torch.Tensor,
                        rewards: torch.Tensor,
                        next_states: torch.Tensor,
                        terminated: torch.Tensor,
                        truncated: torch.Tensor,
                        infos: Any,
                        timestep: int,
                        timesteps: int,
                        env,
                        alive_mask: torch.Tensor = None) -> None:
        """Record an environment transition in memory

        :param states: Observations/states of the environment used to make the decision
        :type states: torch.Tensor
        :param actions: Actions taken by the agent
        :type actions: torch.Tensor
        :param rewards: Instant rewards achieved by the current actions
        :type rewards: torch.Tensor
        :param next_states: Next observations/states of the environment
        :type next_states: torch.Tensor
        :param terminated: Signals to indicate that episodes have terminated
        :type terminated: torch.Tensor
        :param truncated: Signals to indicate that episodes have been truncated
        :type truncated: torch.Tensor
        :param infos: Additional information about the environment
        :type infos: Any type supported by the environment
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        # note this is where I can add tracking stuff
        eval_mode = alive_mask is not None
        
        if not eval_mode and self.memory is not None:
            self._current_next_states = next_states

            # reward shaping
            if self._rewards_shaper is not None:
                rewards = self._rewards_shaper(rewards, timestep, timesteps)

            # compute values
            values, _, _ = self.value.act({"states": self._state_preprocessor(states)}, role="value")
            values = self._value_preprocessor(values, inverse=True)

            # time-limit (truncation) boostrapping
            if self._time_limit_bootstrap:
                rewards += self._discount_factor * values * truncated

            # storage transition in memory
            self.memory.add_samples(states=states, actions=actions, rewards=rewards, next_states=next_states,
                                    terminated=terminated, truncated=truncated, log_prob=self._current_log_prob, values=values)
            for memory in self.secondary_memories:
                memory.add_samples(states=states, actions=actions, rewards=rewards, next_states=next_states,
                                   terminated=terminated, truncated=truncated, log_prob=self._current_log_prob, values=values)
        
        #print('eval mode:', 'on' if eval_mode else 'off')
        if self.write_interval > 0 or eval_mode:
            # compute the cumulative sum of the rewards and timesteps
            if self._cumulative_rewards is None:
                self._cumulative_rewards = torch.zeros_like(rewards, dtype=torch.float32)
                self._cumulative_timesteps = torch.zeros_like(rewards, dtype=torch.int32)
                # create bins for the termination types
                self.m4_returns = {}
                self.count_stats = {}
                self.old_rewards = {}

                #print('Termination Keys')
                for info_key in infos.keys():
                    if not (type(infos[info_key]) == dict):
                        continue
                    for key in infos[info_key].keys():
                        #print(f"\t{key}")
                        if key.startswith("Episode_Termination"):
                            #print(f'\t{key}')
                            self.count_stats[key] = torch.zeros(size=(1, ), device=states.device)
                        elif key.startswith("Episode_Reward"):
                            self.m4_returns[key] = torch.zeros(size=(states.shape[0],1), device=states.device)
                            self.old_rewards[key] = torch.zeros(size=(states.shape[0],1), device=states.device)
                        else:
                            self.m4_returns[key] = torch.zeros(size=(states.shape[0],1), device=states.device)
                            
            # this is a less efficent way to get the termination conditions, but isaac lab api has some issues
            # with not updating those buffers correctly so this is more accurate
            for big_key in infos.keys():
                for k, v in infos[big_key].items():
                    #print("Manager:", env.unwrapped.reward_manager._episode_sums["keypoint_baseline"])
                    #print(f'\t\t{k}:{v}')
                    if k in self.m4_returns.keys():
                        rew = v
                        if 'Reward' in k:
                            rew = torch.unsqueeze(env.unwrapped.reward_manager._episode_sums[k.split("/")[-1]], 1).clone()
                            if eval_mode:
                                self.m4_returns[k] = (rew * alive_mask)
                            else:
                                self.m4_returns[k] = rew
                        else:
                            if eval_mode:
                                self.m4_returns[k] += (rew * alive_mask)
                            else:
                                self.m4_returns[k] += rew
                    else: # it is a count stats key 
                        key = k.split("/")[-1]
                        if eval_mode:
                            self.count_stats[k] += torch.sum( 
                                torch.logical_and(
                                    alive_mask.T, 
                                    env.unwrapped.termination_manager.get_term(key)
                                )
                            )
                        else:
                            self.count_stats[k] += torch.sum(env.unwrapped.termination_manager.get_term(key))

            # handle reward 
            prefix = "Training"
            if eval_mode:
                prefix = "Eval"

            # record data
            self.tracking_data[prefix + " Reward / Instantaneous reward (max)"].append(torch.max(rewards).item())
            self.tracking_data[prefix + " Reward / Instantaneous reward (min)"].append(torch.min(rewards).item())
            self.tracking_data[prefix + " Reward / Instantaneous reward (mean)"].append(torch.mean(rewards).item())

            # handle force data
            self.tracking_data[prefix + " Smoothness / Force (max)"].append(
                torch.max(infos['smoothness']['Smoothness / Damage Force']).item()
            )
            self.tracking_data[prefix + " Smoothness / Force (mean)"].append(
                torch.mean(infos['smoothness']['Smoothness / Damage Force']).item()
            )
            for k, v in self.m4_returns.items():
                #print(k, "\t", v[0].item())
                if 'Force' in k or "Torque" in k:
                    continue
                elif 'Episode_Reward/' in k:
                    # in log IsaacLab returns the sum of each reward type, to get avg 
                    # reward we have to subtract the previous return
                    idx = k.index("/") + 1 
                    my_k = prefix + k[:idx]
                    my_k = my_k.replace('Episode_Reward/', " Reward /")
                    
                    step_vals = v.clone() - self.old_rewards[k]
                    #if 'baseline' in k:
                    #    print(step_vals[0].item(), v[0].item(), self.old_rewards[k][0].item())
                    #if "dist_to_goal" in k[idx:]:
                    #    print(step_vals[0], v[0], self.old_rewards[k][0])
                    self.track_data(f'{my_k + " Step " + k[idx:]} (mean)', torch.mean(step_vals).item())
                    self.track_data(f'{my_k + " Step " + k[idx:]} (max)', torch.max(step_vals).item())
                    self.track_data(f'{my_k + " Step " + k[idx:]} (min)', torch.min(step_vals).item())

                    self.old_rewards[k] = v.clone()

                else:
                    idx = k.index("/") + 1 
                    my_k = prefix + k[:idx]
                    my_k = my_k.replace('Episode_Reward/', " Reward /")
                    my_k = my_k.replace("Smoothness", " Smoothness")
                    #tot = torch.sum(v, dim=0)
                    #stp_avg = tot / self._cumulative_timesteps
                    self.track_data(f'{my_k + " Step " + k[idx:]} (mean)', torch.mean(v).item())
                    self.track_data(f'{my_k + " Step " + k[idx:]} (max)', torch.max(v).item())
                    self.track_data(f'{my_k + " Step " + k[idx:]} (min)', torch.min(v).item())

            if eval_mode:
                self._cumulative_rewards += alive_mask * rewards
                self._cumulative_timesteps[alive_mask] += 1
                mask_update = ~torch.logical_or(terminated, truncated)
                alive_mask *= mask_update

            else:
                self._cumulative_rewards.add_(rewards)
                self._cumulative_timesteps.add_(1)
                mask_update = ~torch.logical_or(terminated, truncated)
                for k,v in self.m4_returns.items():
                    if "Reward" in k:
                        self.old_rewards[k] *= mask_update

            

            finished_episodes = (terminated + truncated).nonzero(as_tuple=False)
            if finished_episodes.numel() and not eval_mode:
                # storage cumulative rewards and timesteps
                self._track_rewards.extend(self._cumulative_rewards[finished_episodes][:, 0].reshape(-1).tolist())
                self._track_timesteps.extend(self._cumulative_timesteps[finished_episodes][:, 0].reshape(-1).tolist())

                # reset the cumulative rewards and timesteps
                self._cumulative_rewards[finished_episodes] = 0
                self._cumulative_timesteps[finished_episodes] = 0
                for k, v in self.m4_returns.items():
                    v[finished_episodes] *= 0
                
                for k,v in self.old_rewards.items():
                    v[finished_episodes] *= 0
                
        return alive_mask
    
    def reset_tracking(self):
        if self._cumulative_rewards is None:
            return
        self._cumulative_rewards *= 0
        self._cumulative_timesteps *= 0
        for k, v in self.m4_returns.items():
            v *= 0

        self._track_rewards.clear()
        self._track_timesteps.clear() 
            

