import copy
from typing import Dict

import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import gymnasium.spaces.utils
import torch
from wrappers.DMP.discrete_dmp import DiscreteDMP
from wrappers.DMP.cs import CS

import matplotlib.pyplot as plt

class DMPObservationWrapper(gym.ObservationWrapper):
    """
        Takes the time series information in observation and replaces it with a set of 
        weights from a DMP that represent the trajectory 


    """
    def __init__(
            self, 
            env, 
            num_weights=10, 
            fit_force_data=False,
            sim_dt = 0.01,
            update_dt = 0.1
        )->None:
        #self.unwrapped = env
        super().__init__(env)

        self.num_weights = num_weights
        self.fit_ft = fit_force_data
        
        
        # calculating t values is tricky, I assume
        # that x = 0 is equal to update_dt
        # while using a tau may be the correct way, simpling scaling
        # the time variable keeps the topology of the trajectory without
        # so long as the canaical system has the same number of steps
        self.sim_dt = sim_dt
        self.update_dt = update_dt
        self.dt = self.sim_dt / self.update_dt
        self.dec = int(self.update_dt / self.sim_dt)

        old_shape = self.observation_space['policy'].shape[1]
        new_dim = self.num_weights * 3 * 1 + old_shape - 6 * self.dec * 3 - self.dec
        #print("old:", old_shape, "new:", new_dim)
        self.observation_space['policy'] = Box(low = -np.inf, high= np.inf, shape=(self.num_envs, new_dim))
        self.single_observation_space['policy'] = Box(low = -np.inf, high= np.inf, shape=(new_dim, ))
        
        # define data containers
        self.t = torch.linspace(start=0, end=self.update_dt, steps= int(self.update_dt/self.sim_dt) ) 
        self.y = torch.zeros((self.num_envs, self.dec, 3), device=self.device)
        self.dy = torch.zeros((self.num_envs, self.dec, 3), device=self.device)
        self.ddy = torch.zeros((self.num_envs, self.dec, 3), device=self.device)
        self.ay = torch.zeros((self.num_envs, self.dec, 4), device=self.device)
        self.day = torch.zeros((self.num_envs, self.dec, 3), device=self.device)
        self.dday = torch.zeros((self.num_envs, self.dec, 3), device=self.device)

        # set a ref list to make step func look pretty
        self.unpack_list = [
            ("fingertip_pos", self.y),
            ("fingertip_quat", self.ay),
            ("ee_linvel", self.dy),
            ("ee_angvel", self.day),
            ("ee_linacc", self.ddy),
            ("ee_angacc", self.dday)
        ]

        # define the DMPs
        self.cs = CS(ax=2.5, dt=self.dt, device=self.device)
        self.pos_dmp = DiscreteDMP(
            nRBF=self.num_weights, 
            betaY=12.5/4.0, 
            dt=self.dt, 
            cs=self.cs, 
            num_envs=self.num_envs, 
            num_dims=3,
            device=self.device
        )

        # our new obs (num_weights * dim_per_(ang/pos) * (ang + pos))
        # current last mult is 1 because I'm not considering angular anything
        self.new_obs = torch.zeros( (self.num_envs, new_dim), device=self.device)



    def observation(self, old_obs):
        idx = 0
        for i in range(len(self.unpack_list)):
            var_name, var_ref = self.unpack_list[i]
            dim = 4 if "quat" in var_name else 3
            #print(f"{var_name}: {idx}, {idx + self.dec*dim}")
            var_ref[:,:,:] = old_obs['policy'][:, idx:idx + self.dec*dim].view(
                (self.num_envs, self.dec, dim)
            )
            idx += self.dec * dim
        
        #print(self.y[0,:,0])
        self.pos_dmp.learnWeightsCSDT(self.y, self.dy, self.ddy, self.t)
        #print(self.pos_dmp.ws)
        self.new_obs[:, :(self.num_weights * 3) ] = self.pos_dmp.ws.view(self.num_envs, self.num_weights * 3)
        self.new_obs[:, (self.num_weights * 3 * 1):] = old_obs['policy'][:, (6 * self.dec * 3 + self.dec):]

        ts, z, dz, ddz = self.pos_dmp.rollout(self.y[:,-1,:], self.y[:,0,:], self.dy[:,0,:], self.ddy[:,0,:])


        fig, axs = plt.subplots(self.num_envs, 3)
        fig.set_figwidth(3 / 3 * 1600/96)
        fig.set_figheight(self.num_envs / 4 * 1000/96)
        fig.tight_layout(pad=5.0)
        for i in range(self.num_envs):
            for j in range(3):
                axs[i,j].plot(self.t.cpu(), self.y[i,:,j].cpu(), label="Original")
                axs[i,j].plot(ts.cpu()*0.1, z[i,:,j].cpu(), 'r--', label="Fit DMP")
                axs[i,j].set_title(f"DMP (env={i}, dim={j})",  fontsize=20)
                axs[i,j].set(xlabel = 'Time (s)')
                axs[i,j].set(ylabel ='Position (m)')

        plt.show()

        old_obs['policy'] = self.new_obs
        return old_obs
    """
    def reset(self, **kwargs):
        old_obs, info = self.env.reset(**kwargs)
        # weights we want, plus old obs, minus all vars for the histories
        
        if not self.new_obs.size()[1] == new_dim:
            self.new_obs = torch.zeros( (self.num_envs, new_dim), device=self.device)
        
        for i in range(len(self.unpack_list)):
            var_ref, var_name = self.unpack_list[i]
            var_ref = old_obs['policy'][var_name][:, i*self.dec:(i+1)*self.dec].view(
                (self.num_envs, 4 if "quat" in var_name else 3)
            )
        
        self.pos_dmp.learnWeightsCSDT(self.y, self.dy, self.ddy, self.t)
        self.new_obs[:, :self.num_weights * 3 ] = self.pos_dmp.ws.view(self.num_envs, self.num_weights * 3)
        self.new_obs[:, self.num_weights * 3 * 1:] = old_obs[:, (6 * self.dec * 3):]
        return self.new_obs, info
    
    def step(self, action):
        #observation, r, term, trun, info = self.unwrapped.step(action)
        old_obs, r, term, trun, info = self.env.step(action)

        for i in range(len(self.unpack_list)):
            var_ref, var_name = self.unpack_list[i]
            var_ref = old_obs['policy'][var_name][:, i*self.dec:(i+1)*self.dec].view(
                (self.num_envs, 4 if "quat" in var_name else 3)
            )
        
        self.pos_dmp.learnWeightsCSDT(self.y, self.dy, self.ddy, self.t)
        self.new_obs[:, :self.num_weights * 3 ] = self.pos_dmp.ws.view(self.num_envs, self.num_weights * 3)
        self.new_obs[:, self.num_weights * 3 * 1:] = old_obs[:, (6 * self.dec * 3):]
        return self.new_obs, r, term, trun, info
    """