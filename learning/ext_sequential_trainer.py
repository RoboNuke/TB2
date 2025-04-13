from typing import List, Optional, Union

import copy
import sys
import tqdm

import torch

from skrl.agents.torch import Agent
from skrl.envs.wrappers.torch import Wrapper
from skrl.trainers.torch import Trainer

from agents.agent_list import AgentList

import os
import sys
import warnings
# stolen from https://github.com/elle-miller/skrl_testing/blob/main/skrl_testing/utils/skrl/sequential.py#L149

# [start-config-dict-torch]
EXT_SEQUENTIAL_TRAINER_DEFAULT_CONFIG = {
    "timesteps": 100000,            # number of timesteps to train for
    "headless": False,              # whether to use headless mode (no rendering)
    "disable_progressbar": False,   # whether to disable the progressbar. If None, disable on non-TTY
    "close_environment_at_exit": True,   # whether to close the environment on normal program termination
    "environment_info": "episode",  # key used to get and log environment info
}
# [end-config-dict-torch]


class ExtSequentialTrainer(Trainer):
    def __init__(self,
                 env: Wrapper,
                 agents: Union[Agent, List[Agent]],
                 agents_scope: Optional[List[int]] = None,
                 cfg: Optional[dict] = None) -> None:
        """Sequential trainer

        Train agents sequentially (i.e., one after the other in each interaction with the environment)

        :param env: Environment to train on
        :type env: skrl.envs.wrappers.torch.Wrapper
        :param agents: Agents to train
        :type agents: Union[Agent, List[Agent]]
        :param agents_scope: Number of environments for each agent to train on (default: ``None``)
        :type agents_scope: tuple or list of int, optional
        :param cfg: Configuration dictionary (default: ``None``).
                    See SEQUENTIAL_TRAINER_DEFAULT_CONFIG for default values
        :type cfg: dict, optional
        """
        _cfg = copy.deepcopy(EXT_SEQUENTIAL_TRAINER_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})

        agents_scope = agents_scope if agents_scope is not None else []
        
        super().__init__(env=env, agents=agents, agents_scope=agents_scope, cfg=_cfg)
        
        self.abs_agent = AgentList(agents, self.agents_scope, cfg=cfg)
        
        # init agents
        self.abs_agent.init(trainer_cfg=self.cfg)
        self.training_timestep = 0


    def train(self, train_timesteps, vid_env) -> None:
        """Train the agents sequentially for train timesteps

        This method executes the following steps in loop:

        - Pre-interaction (sequentially)
        - Compute actions (sequentially)
        - Interact with the environments
        - Render scene
        - Record transitions (sequentially)
        - Post-interaction (sequentially)
        - Reset environments
        """
        # set running mode
        self.abs_agent.set_running_mode("train")

        # non-simultaneous agents
        assert self.env.num_agents == 1, "Does not support multiagent training"

        self.abs_agent.reset_tracking()
        self.env.unwrapped.evaluating = False
        # reset env 
        states, infos = self.env.unwrapped.reset() # reseting unwrapped seems to ensure full reset idk
        states, infos = self.env.reset()
        

        #self.abs_agent.reset_memory()
        self.abs_agent._rollout = 0
        # self.agents._cumulative_rewards = None
        # self.agents._cumulative_timesteps = None
        # allocate some memeroy
        term_man = self.env.unwrapped.termination_manager
        term_cond = term_man.active_terms
        term_dist = {}
        for con in term_cond:
            term_dist[con] = torch.zeros_like(term_man.get_term(con))
        
        rew_man = self.env.unwrapped.reward_manager
        rew_types = rew_man._episode_sums.keys()
        rew_dist = {}
        for con in rew_types:
            rew_dist[con] = torch.zeros_like( torch.unsqueeze(rew_man._episode_sums[con], 1)) 


        train_start = self.training_timestep
        train_pause = self.training_timestep + train_timesteps
        
        for timestep in tqdm.tqdm(range(train_start, train_pause), disable=self.disable_progressbar, file=sys.stdout):
            
            
            # pre-interaction
            self.abs_agent.pre_interaction(timestep=timestep, timesteps=self.timesteps)

            # compute actions
            with torch.no_grad():
                
                actions = self.abs_agent.act(
                    states, 
                    timestep=timestep, 
                    timesteps=self.timesteps
                )[0] # we take only the sampled action
                
                # step the environments
                #print("before len buf:\t", self.env.unwrapped.episode_length_buf)
                #print("before timout:", self.env.unwrapped.termination_manager.get_term("time_out"))
                next_states, rewards, terminated, truncated, infos = self.env.step(actions)
                #print("after timout:", self.env.unwrapped.termination_manager.get_term("time_out"))
                #print("termed:\t", self.env.unwrapped.termination_manager.terminated)
                #print("time_outs:\t", self.env.unwrapped.termination_manager.time_outs)
                #print("after len buf:\t", self.env.unwrapped.episode_length_buf)
                infos['log']['Episode_Termination/time_out'] = truncated.sum()
                if vid_env is not None and vid_env.is_recording():
                    self.env.cfg.recording = True
                    
                #print(infos['smoothness'])
                # render scene
                if not self.headless:
                    print("rendering")
                    self.env.render()

                # get specific reward and termination data
                for rew_type in rew_types:
                    rew_dist[rew_type] = torch.unsqueeze(rew_man._episode_sums[rew_type], 1).clone()

                for con in term_cond:
                    term_dist[con] = term_man.get_term(con)

                # record the environments' transitions
                self.abs_agent.record_transition(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    infos=infos,
                    timestep=timestep,
                    reward_dist = rew_dist,
                    term_dist = term_dist,
                    timesteps=self.timesteps
                )
                


                # log environment info
                if self.environment_info in infos:
                    for k, v in infos[self.environment_info].items():
                        print(k, v)
                        if isinstance(v, torch.Tensor) and v.numel() == 1:
                            self.abs_agent.track_data(f"Info / {k}", v.item())

            # post-interaction
            self.abs_agent.post_interaction(timestep=timestep, timesteps=self.timesteps)
            

            # reset environments
            if self.env.num_envs > 1:
                states = next_states
            else:
                if terminated.any() or truncated.any():
                    with torch.no_grad():
                        states, infos = self.env.reset()
                else:
                    states = next_states

            self.training_timestep += 1

    def eval(self, global_step, vid_env, record=False) -> None:
        """Evaluate the agents sequentially

        This method executes the following steps in loop:

        - Compute actions (sequentially)
        - Interact with the environments
        - Render scene
        - Reset environments
        """
        # set running mode
        self.abs_agent.set_running_mode("eval")

        # non-simultaneous agents
        assert self.env.num_agents == 1, "Does not support multiagent training"

        print("init training timestep:", self.training_timestep)

        self.abs_agent.reset_tracking()
        # reset env - unwrapped required because skrl does not call reset functions above it
        states, infos = self.env.unwrapped.reset()
        states, infos = self.env.reset()

        self.env.unwrapped.evaluating = True

        ep_length = self.env.env.max_episode_length #- 1
                
        term_man = self.env.unwrapped.termination_manager
        term_cond = term_man.active_terms
        term_dist = {}
        for con in term_cond:
            term_dist[con] = torch.zeros_like(term_man.get_term(con))
        
        rew_man = self.env.unwrapped.reward_manager
        rew_types = rew_man._episode_sums.keys()
        rew_dist = {}
        for con in rew_types:
            rew_dist[con] = torch.zeros_like( torch.unsqueeze(rew_man._episode_sums[con], 1)) 


        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            alive_mask = torch.ones(size=(states.shape[0], 1), device=states.device, dtype=bool)
            for timestep in tqdm.tqdm(range(self.initial_timestep, ep_length), disable=self.disable_progressbar, file=sys.stdout):
                
                # compute actions
                with torch.no_grad():
                    actions = self.abs_agent.act(
                        states, 
                        timestep=timestep, 
                        timesteps=self.timesteps
                    )[-1]['mean_actions'] # this makes the policy deterministic (no sampling)
                    
                    actions[~alive_mask[:,0], :] *= 0.0
                    # step the environments
                    next_states, rewards, terminated, truncated, infos = self.env.step(actions)
                    
                    if vid_env is not None and vid_env.is_recording():
                        self.env.cfg.recording = True
                    
                    mask_update = 1 - torch.logical_or(terminated, truncated).float()
                    
                    self.env.unwrapped.common_step_counter -= 1
                    
                    # get specific reward and termination data
                    for rew_type in rew_types:
                        rew_dist[rew_type] = torch.unsqueeze(rew_man._episode_sums[rew_type], 1).clone()

                    for con in term_cond:
                        term_dist[con] = term_man.get_term(con)
                    
                    alive_mask = self.abs_agent.record_transition(
                        states=states,
                        actions=actions,
                        rewards=rewards,
                        next_states=next_states,
                        terminated=terminated,
                        truncated=truncated,
                        infos=infos,
                        timestep=timestep,
                        timesteps=self.timesteps,
                        reward_dist = rew_dist,
                        term_dist = term_dist,
                        alive_mask = alive_mask
                    )

                    #alive_mask *= mask_update
                    
                    # render scene
                    if not self.headless:
                        self.env.render()
                    
                    #self.abs_agent.post_interaction(timestep=timestep, timesteps=self.timesteps, eval=True)

                    # reset environments
                    if self.env.num_envs > 1:
                        states = next_states
                    else:
                        if terminated.any() or truncated.any():
                            with torch.no_grad():
                                states, infos = self.env.reset()
                        else:
                            states = next_states
                            
        self.abs_agent.write_tracking_data(self.training_timestep, self.timesteps, eval=True)