import torch
import torch.nn as nn
import numpy as np

from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
import math

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class NatureCNN(nn.Module):
    def __init__(self, 
            obs_size,
            with_state=True, 
            with_rgb=False, 
            force_type=None
        ):#, nn.ReLU=nn.ReLU):
        super().__init__()
        with_force = force_type is not None

        # TODO: Handle n-step
        #if "-step" in force_type:
        #    force_type="FFN"

        extractors = {}
        self.out_features = 0
        feature_size = 256
        if with_rgb:
            #TODO: This doesn't work
            """
            in_channels=sample_obs["rgb"].shape[-1]
            image_size=(sample_obs["rgb"].shape[1], sample_obs["rgb"].shape[2])


            # here we use a NatureCNN architecture to process images, but any architecture is permissble here
            cnn = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=32,
                    kernel_size=8,
                    stride=4,
                    padding=0,
                ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0
                ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0
                ),
                nn.ReLU(),
                nn.Flatten(),
            )

            # to easily figure out the dimensions after flattening, we pass a test tensor
            with torch.no_grad():
                n_flatten = cnn(sample_obs["rgb"].float().permute(0,3,1,2).cpu()).shape[1]
                fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
            extractors["rgb"] = nn.Sequential(cnn, fc)
            self.out_features += feature_size
            """
            pass
        #self.cat_state_force = False
        if with_state and with_force:
            #self.cat_state_force = True
            state_size = obs_size
            if force_type == "FFN":
                force_size = 0
            else:
                raise NotImplementedError(f"Unexpected force type:{force_type}")
            
            extractors['state'] = nn.Sequential(
                nn.Linear(state_size + force_size, 256),
                nn.ReLU()
            )
            self.out_features += 256

        if with_state and not with_force:
            # for state data we simply pass it through a single linear layer
            #extractors["state"] = nn.Sequential(
            
            self.feat_extractor = nn.Sequential(
                nn.Linear(obs_size, 256),
                nn.ReLU()
            )
            self.out_features += 256

        elif with_force and not with_state:
            if force_type == "FFN":
                force_size = obs_size
                extractors["force"] = nn.Sequential(
                    nn.Linear(force_size, 256),
                    nn.ReLU()
                )
                self.out_features += 256
                
            elif force_type == "1D-CNN":
                raise NotImplementedError(f"1D-CNN not implemented yet")
            else:
                raise NotImplementedError(f"Unexpected force type:{force_type}")
            
        #self.extractors = nn.ModuleDict(extractors)

    def forward(self, observations) -> torch.Tensor:
        #print(observations)
        #encoded_tensor_list = []
        # self.extractors contain nn.Modules that do all the processing.
        """
        #for key, extractor in self.extractors.items():
        #    obs = observations[key]
            
            TODO: Handle RGB + Force observations seperately
            if self.cat_state_force and (key == 'state'):
                f_obs = torch.tanh( observations['force'] * 0.0011)
                s_obs = observations['state']
                obs = torch.cat([s_obs, f_obs], dim=1) 
                encoded_tensor_list.append(extractor(obs))
            elif not self.cat_state_force:
                if key == "rgb":
                    obs = obs.float().permute(0,3,1,2)
                    obs = obs / 255
                if key == 'force':
                    obs = torch.tanh( obs.float() * 0.0011 )
            
        #    encoded_tensor_list.append(extractor(obs))
        #return torch.cat(encoded_tensor_list, dim=1)
        """
        return self.feat_extractor(observations['states'])

class BroNetLayer(nn.Module):
    def __init__(self, size, device):
        super().__init__()

        self.path = nn.Sequential(
            layer_init(nn.Linear(size, size)),
            nn.LayerNorm(size),
            #nn.GLU(),
            #CustomGLU(size),
            nn.ReLU(),
            layer_init(nn.Linear(size, size)),
            nn.LayerNorm(size)
        ).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.path(res)
        return out + res

class BroNet(nn.Module):
    def __init__(self, n, in_size, out_size, latent_size, device, tanh_out=False):
        super().__init__()
        self.layers = []
        self.n = n
        self.input = nn.Sequential(
            layer_init(nn.Linear(in_size, latent_size)),
            nn.LayerNorm(latent_size),
            #nn.GLU(),
            #CustomGLU(size),
            nn.ReLU(),
        ).to(device)

        self.layers = nn.ModuleList([BroNetLayer(latent_size, device) for i in range(self.n)])

        if not tanh_out:
            self.output = layer_init(nn.Linear(latent_size, out_size)).to(device)
        else:
            self.output = nn.Sequential(
                layer_init(nn.Linear(latent_size, out_size)).to(device),
                nn.Tanh()
            ).to(device)

    def forward(self, x):
        out =  self.input(x)
        for i in range(self.n):
            out = self.layers[i](out)
        out = self.output(out)
        return out
    
class BroAgent(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, 
            observation_space,
            action_space,
            device, 
            act_init_std = 0.60653066, # -0.5 value used in maniskill

            force_type=None, 
            critic_n = 1, 
            actor_n = 2,
            critic_latent=128,
            actor_latent=512,

            clip_actions=False,
            clip_log_std=True, 
            min_log_std=-20, 
            max_log_std=2, 
            reduction="sum"
        ):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self, clip_actions)
        self.feature_net = NatureCNN(
            obs_size=self.num_observations, 
            with_state=True,
            with_rgb=False,
            force_type=force_type
        )#, act_func=nn.ReLU)

        #print("feat net\n", self.feature_net)
        in_size = self.feature_net.out_features
        
        self.critic = BroNet(
            critic_n, 
            in_size, 
            1, 
            critic_latent, 
            device, 
            tanh_out=False
        )
        
        #self.actors = [BroNet(actor_n, in_size, self.act_size, actor_latent, device, tanh_out=True) for i in range(tot_actors)]
        self.actor_mean = BroNet(
            actor_n, 
            in_size, 
            self.num_actions, 
            actor_latent, 
            device, 
            tanh_out=True
        )

        layer_init(self.actor_mean.output[-2], std=0.01*np.sqrt(2)) 
 
        self.actor_logstd = nn.Parameter(
            torch.ones(1, self.num_actions) * math.log(act_init_std)
        )

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        if role == "policy":
            self._shared_output = self.feature_net(inputs)
            action_mean = self.actor_mean(self._shared_output)
            return action_mean, self.actor_logstd.expand_as(action_mean), {}
        elif role == "value":
            shared_output = self.feature_net(inputs) if self._shared_output is None else self._shared_output
            self._shared_output = None
            return self.critic(shared_output), {}
        

class BroCritic(DeterministicMixin, Model):
    def __init__(self, 
            observation_space,
            action_space,
            device, 

            force_type=None, 
            critic_n = 1, 
            critic_latent=128,

            clip_actions=False,
        ):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)
        self.feature_net = NatureCNN(
            obs_size=self.num_observations, 
            with_state=True,
            with_rgb=False,
            force_type=force_type
        )#, act_func=nn.ReLU)

        #print("feat net\n", self.feature_net)
        in_size = self.feature_net.out_features
        
        self.critic = BroNet(
            critic_n, 
            in_size, 
            1, 
            critic_latent, 
            device, 
            tanh_out=False
        )

    def act(self, inputs, role):
        return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        return self.critic(self.feature_net(inputs)), {}
    

class BroActor(GaussianMixin, Model):
    def __init__(self, 
            observation_space,
            action_space,
            device, 
            act_init_std = 0.60653066, # -0.5 value used in maniskill

            force_type=None, 
            actor_n = 2,
            actor_latent=512,

            clip_actions=False,
            clip_log_std=True, 
            min_log_std=-20, 
            max_log_std=2, 
            reduction="sum"
        ):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        self.feature_net = NatureCNN(
            obs_size=self.num_observations, 
            with_state=True,
            with_rgb=False,
            force_type=force_type
        )#, act_func=nn.ReLU)

        #print("feat net\n", self.feature_net)
        in_size = self.feature_net.out_features
        
        self.actor_mean = BroNet(
            actor_n, 
            in_size, 
            self.num_actions, 
            actor_latent, 
            device, 
            tanh_out=True
        )

        layer_init(self.actor_mean.output[-2], std=0.01*np.sqrt(2)) 
 
        self.actor_logstd = nn.Parameter(
            torch.ones(1, self.num_actions) * math.log(act_init_std)
        )

    def act(self, inputs, role):
        return GaussianMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        action_mean = self.actor_mean(self.feature_net(inputs))
        return action_mean, self.actor_logstd.expand_as(action_mean), {}