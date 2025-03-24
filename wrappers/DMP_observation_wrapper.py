import copy
from typing import Dict

import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import gymnasium.spaces.utils
import torch
from wrappers.DMP.discrete_dmp import DiscreteDMP
from wrappers.DMP.cs import CS
from wrappers.DMP.quaternion_dmp import QuaternionDMP

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
            update_dt = 0.1,
            save_fit = False
        )->None:
        #self.unwrapped = env
        super().__init__(env)

        self.num_weights = num_weights
        self.fit_ft = fit_force_data
        self.save_fit = True
        
        self.display_fit = True
        # calculating t values is tricky, I assume
        # that x = 0 is equal to update_dt
        # while using a tau may be the correct way, simpling scaling
        # the time variable keeps the topology of the trajectory without
        # so long as the canaical system has the same number of steps
        self.sim_dt = sim_dt
        self.update_dt = update_dt
        self.dt = self.sim_dt / self.update_dt
        self.dec = int(self.update_dt / self.sim_dt)

        #old_shape = self.observation_space['policy'].shape[1]
        new_dim = self.num_weights * 6 #3 * 1 + old_shape - 6 * self.dec * 3 - self.dec
        #print("old:", old_shape, "new:", new_dim)
        self.observation_space['policy'] = Box(low = -np.inf, high= np.inf, shape=(self.num_envs, new_dim))
        self.single_observation_space['policy'] = Box(low = -np.inf, high= np.inf, shape=(new_dim, ))
        
        # define data containers
        # note: you add one to include the starting point. This is the last location the policy was at in the last timestep
        
        self.t = torch.linspace(start=0, end=self.update_dt, steps= int(self.update_dt/self.sim_dt)+1 ) 
        self.y = torch.zeros((self.num_envs, self.dec+1, 3), device=self.device)
        self.dy = torch.zeros((self.num_envs, self.dec+1, 3), device=self.device)
        self.ddy = torch.zeros((self.num_envs, self.dec+1, 3), device=self.device)
        self.ay = torch.zeros((self.num_envs, self.dec+1, 4), device=self.device)
        self.ay[:,:,0] = 1.0
        self.day = torch.zeros((self.num_envs, self.dec+1, 4), device=self.device)
        self.dday = torch.zeros((self.num_envs, self.dec+1, 4), device=self.device)

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
        self.cs = CS(
            ax=2.5, 
            dt=self.dt, 
            device=self.device
        )

        self.pos_dmp = DiscreteDMP(
            nRBF=self.num_weights, 
            betaY=12.5/4.0, 
            dt=self.dt, 
            cs=self.cs, 
            num_envs=self.num_envs, 
            num_dims=3,
            device=self.device
        )


        self.ang_dmp = QuaternionDMP(
            nRBF=self.num_weights,
            betaY=12/4,
            dt = self.dt,
            cs = CS(
                    ax=1, 
                    dt=self.dt, 
                    device=self.device
            ),
            num_envs= self.num_envs,
            device = self.device
        )

        # our new obs (num_weights * dim_per_(ang/pos) * (ang + pos))
        # current last mult is 1 because I'm not considering angular anything
        self.new_obs = torch.zeros( (self.num_envs, new_dim), device=self.device)
        if self.save_fit or self.display_fit:
            self.fig, self.axs = plt.subplots(self.num_envs, 4)
            self.fig.set_figwidth(3 / 3 * 1600/96)
            self.fig.set_figheight(self.num_envs / 4 * 1000/96)
            self.fig.tight_layout(pad=5.0)
            self.start_time = 0.0



    def observation(self, old_obs):
        idx = 0
        for i in range(len(self.unpack_list)):
            var_name, var_ref = self.unpack_list[i]
            dim = 4 if "quat" in var_name else 3
            #print(f"{var_name}: {idx}, {idx + self.dec*dim}")
            if "ang" in var_name:
                var_ref[:,1:,1:] = old_obs['policy'][:, idx:idx + self.dec*dim].view(
                    (self.num_envs, self.dec, dim)
                )
            else:
                var_ref[:,1:,:] = old_obs['policy'][:, idx:idx + self.dec*dim].view(
                    (self.num_envs, self.dec, dim)
                )
            idx += self.dec * dim
        
        #print(self.y[0,:,0])
        self.pos_dmp.learnWeightsCSDT(self.y, self.dy, self.ddy, self.t)
        self.ang_dmp.learnWeightsCSDT(self.ay, self.day, self.dday, self.t)
        #print(self.ay)
        #print(torch.linalg.norm(self.ay,dim=2))
        self.new_obs[:, :(self.num_weights * 3) ] = self.pos_dmp.ws.view(self.num_envs, self.num_weights * 3)
        self.new_obs[:, (self.num_weights * 3):] = self.ang_dmp.w.reshape(self.num_envs, self.num_weights * 3) 
        #print(self.new_obs)

        # move final obs to init position
        for i in range(len(self.unpack_list)):
            var_name, var_ref = self.unpack_list[i]
            var_ref[:,0,:] = var_ref[:,-1,:]

        if (self.save_fit or self.display_fit) and self.start_time < 4.9:
            ts, z, dz, ddz = self.pos_dmp.rollout(
                self.y[:,-1,:], 
                self.y[:,0,:], 
                self.dy[:,0,:], 
                self.ddy[:,0,:]
            )
            ts, az, daz, ddaz = self.ang_dmp.rollout(
                self.ay[:,-1,:][:,None,:], 
                self.ay[:,0,:][:,None,:], 
                self.day[:,0,:][:,None,:], 
                self.dday[:,0,:][:,None,:]
            )
            
            ts += self.start_time
            for i in range(self.num_envs):
                for j in range(3):
                    self.axs[i,j].plot(self.t.cpu()+self.start_time, self.y[i,:,j].cpu(), label="Original")
                    self.axs[i,j].plot(self.t.cpu()+self.start_time, z[i,:,j].cpu(), 'r--', label="Fit DMP")
                    self.axs[i,j].set_title(f"DMP (env={i}, dim={j})",  fontsize=20)
                    self.axs[i,j].set(xlabel = 'Time (s)')
                    self.axs[i,j].set(ylabel ='Position (m)')

                self.axs[i,3].plot(self.t.cpu()+self.start_time, self.ay[i,:,0].cpu(), label="w", color='m')
                self.axs[i,3].plot(self.t.cpu()+self.start_time, self.ay[i,:,1].cpu(), label="x", color='r')
                self.axs[i,3].plot(self.t.cpu()+self.start_time, self.ay[i,:,2].cpu(), label="y", color='g')
                self.axs[i,3].plot(self.t.cpu()+self.start_time, self.ay[i,:,3].cpu(), label="z", color='b')

                self.axs[i,3].plot(self.t.cpu()+self.start_time, az[i,:,0].cpu(), label="w - Fit", color='k', linestyle=":")
                self.axs[i,3].plot(self.t.cpu()+self.start_time, az[i,:,1].cpu(), label="x - Fit", color='k', linestyle=":")
                self.axs[i,3].plot(self.t.cpu()+self.start_time, az[i,:,2].cpu(), label="y - Fit", color='k', linestyle=":")
                self.axs[i,3].plot(self.t.cpu()+self.start_time, az[i,:,3].cpu(), label="z - Fit", color='k', linestyle=":")
            #plt.show()
            #print("times:", self.start_time, self.start_time + self.update_dt)
            self.start_time += self.update_dt

        old_obs['policy'] = self.new_obs
        #print("Done!")
        return old_obs
    
    def reset(self, **kwargs):
        """Reset the environment using kwargs and then starts recording if video enabled."""
        
        if self.save_fit:
            plt.savefig("/home/hunter/Fit.png")
        if self.display_fit:
            plt.show()
        if self.save_fit or self.display_fit:
            plt.clf()
            self.fig, self.axs = plt.subplots(self.num_envs, 4)
            self.fig.set_figwidth(3 / 3 * 1600/96)
            self.fig.set_figheight(self.num_envs / 4 * 1000/96)
            self.fig.tight_layout(pad=5.0)
            self.start_time = 0.0
        
        #print("\n\n\n\nreset\n\n\n\n\n")
        #assert 1 == 0
        observations, info = super().reset(**kwargs)



        return observations, info