from skrl.agents.torch import Agent
from typing import Any, Mapping, Optional, Tuple, Union
from skrl.memories.torch import Memory
from skrl.models.torch import Model

import gym
import gymnasium

import torch


class ToyAgent(Agent):
    def __init__(self,
                 models: Mapping[str, Model],
                 memory: Optional[Union[Memory, Tuple[Memory]]] = None,
                 observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                 action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                 device: Optional[Union[str, torch.device]] = None,
                 cfg: Optional[dict] = None) -> None:
        
        super().__init__(models, memory, observation_space, action_space, device, cfg)
        self.action = torch.zeros_like(action_space)

    def init(self, trainer_cfg):
        super().init(trainer_cfg)

    def track_data(self, tag: str, value: float) -> None:
        super().track_data(tag, value)

    def act(self,
        states: torch.Tensor,
        timestep: int,
        timesteps: int) -> torch.Tensor:
        print("Act")
        return self.action

    def pre_interaction(self, timestep: int, timesteps: int) -> None:
        print("Pre-interaction")
    
    def post_interaction(self, timestep: int, timesteps: int, eval: bool = False) -> None:
        print("Post-Interaction")


class PrintActivity(gymnasium.Wrapper):
    def step(self, actions):
        #print("step")
        next_states, rewards, terminated, truncated, infos = super().step(actions)
        if terminated.any():
            print("Terminated:", terminated)
            print(infos)
        if truncated.any():
            print("truncated:", truncated)
            print(infos)
        return next_states, rewards, terminated, truncated, infos
    
    def reset(self):
        print("\n\n\nreset\n\n\n")
        obs, info = super().reset()
        print(obs, info)
        return obs, info
