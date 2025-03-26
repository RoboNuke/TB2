import unittest
import torch
from typing import Any, Mapping, Optional, Tuple, Union

from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.memories.torch import RandomMemory

import gym
import gymnasium
from gymnasium.spaces import Box

import numpy as np
import torch.multiprocessing as mp

from agents.mp_agent import MPAgent
from models.bro_model import BroAgent
from agents.agent_list import AgentList

class DummyAgent():
    def __init__(
            self,
            models: Mapping[str, Model],
            memory: Optional[Union[Memory, Tuple[Memory]]] = None,
            observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
            action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
            num_envs: int = 256,
            device: Optional[Union[str, torch.device]] = None,
            cfg: Optional[dict] = None
    ) -> None:
        self.device = device
        self.training = False
        self.idx = cfg['idx']
        self.memory = memory
        self.models = models
        self.cfg = cfg

        self.policy = self.models.get("policy", None)
        self.value = self.models.get("value", None)

        self.action = self.idx * torch.ones((num_envs, action_space.shape[0]), dtype=torch.float32, device = self.device)
        self.observation = self.idx * torch.ones((num_envs, observation_space.shape[0]), dtype=torch.float32, device = self.device)

    def init(self, trainer_cfg: Optional[Mapping[str, Any]] = None) -> None:
        pass


    def act(self,
            states: torch.Tensor,
            timestep: int,
            timesteps: int) -> torch.Tensor:
        pre_act = self.policy.act({"states": states}, role="policy")
        act = (self.action, pre_act[1], pre_act[2])
        assert states.size() == self.observation.size()
        assert timestep < timesteps
        return act
    
    def pre_interaction(self, timestep: int, timesteps: int) -> None:
        print("called lowest action")
        assert timestep < timesteps

    def record_transition(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: Any,
        timestep: int,
        timesteps: int,
        reward_dist: dict,
        term_dist: dict,
        alive_mask: torch.Tensor = None) -> None:

        if self.memory is not None:
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
        




class TestMPAgent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.num_envs = 20
        cls.num_agents = 4
        cls.device = 'cpu'
        cls.action_space = Box(low = -np.inf, high= np.inf, shape=(5,))
        cls.observation_space = Box(low = -np.inf, high= np.inf, shape=(10,))

        cls.memories = [ RandomMemory(
                        memory_size=50, 
                        num_envs=cls.num_envs // cls.num_agents, 
                        device=cls.device
                    ) for i in range(cls.num_agents)
        ]

        cls.models = {}
        cls.models['policy'] = BroAgent(
            observation_space=cls.observation_space, 
            action_space=cls.action_space,
            device=cls.device
        ) 
        cls.models['value'] = cls.models['policy']

        cls.cfg = {'idx':0, "agent_is_list": False}
        import copy
        cls.agents = [
            DummyAgent(
                models=copy.deepcopy(cls.models),
                memory= cls.memories[i],
                observation_space=cls.observation_space,
                action_space=cls.action_space,
                num_envs=cls.num_envs // cls.num_agents,
                device="cpu",
                cfg={'idx':i, "agent_is_list": False}
            ) 
            for i in range(cls.num_agents)
        ]

        n = int(cls.num_envs/cls.num_agents)
        cls.agent_scope = [[i * n, (i+1) * n] for i in range(cls.num_agents)]      

        cls.cur_agent = None
        mp.set_start_method("spawn")
        #cls.cur_agent = MPAgent(agents=cls.agents, agents_scope=cls.agent_scope )
        #cls.cur_agent.init({"idx":0})


    def tearDown(self):
        if self.cur_agent is not None:
            self.cur_agent.__del__()
        self.cur_agent = None

    def setUp(self):
        self.cur_agent = MPAgent(agents=self.agents, agents_scope=self.agent_scope )
        self.cur_agent.init({"idx":0})


    def test_act(self):
        
        for _ in range(10):
            states = torch.ones((self.num_envs, 10), dtype=torch.float32, device=self.device)
            action = self.cur_agent.act(states, 0, 500)

            sample_action = torch.tensor(self.action_space.sample()).repeat(self.num_envs, 1)
            assert action.dtype == sample_action.dtype, f"Action type is {action.dtype}, but should be {sample_action.dtype}"
            assert action.size() == sample_action.size(), f"Action size is {action.size()}, but should be {sample_action.size()}"
            for k in range(self.num_agents):
                a = self.agent_scope[k][0]
                b = self.agent_scope[k][1]
                assert torch.sum(action[a:b,:]) == k * (b - a) * 5, f"Action {k} sums to {torch.sum(action[a:b,:])} but should be {k * (b - a) * 5}"
        
    def test_preinteraction(self):

        for _ in range(10):
            self.cur_agent.pre_interaction(0,500)

        for que in self.cur_agent.queues:
            assert que.empty(), "Queues are not empty"

    def test_record_transition(self):
        states = torch.ones((self.num_envs, 10), dtype=torch.float32, device=self.device)
        for _ in range(10): 
            self.cur_agent.pre_interaction(0,500)
            self.cur_agent.act(states, 0, 500)

    
