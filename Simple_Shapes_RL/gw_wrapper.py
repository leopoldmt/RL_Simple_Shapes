from gym import Env, ObservationWrapper
from torch.nn import functional as F
from gymnasium.spaces import Box, Dict
import numpy as np
import torch


def get_obs_space(obs_mode):
    if obs_mode == 'gw_v' or obs_mode == 'gw_attr':
        return Box(low=-np.inf, high=np.inf, shape=(12,))
    elif obs_mode == 'attr':
        return Box(low=-1, high=1, shape=(11,))
    elif obs_mode == 'v':
        return Box(low=-np.inf, high=np.inf, shape=(12,))
    

class GWWrapper(ObservationWrapper):
    def __init__(self, env, model, mode):
        super().__init__(env)
        self.model = model
        self.mode = mode
        self.observation_space = get_obs_space(mode)

    def observation(self, obs):
        return self._get_GW(obs, self.mode)
     
    def _get_GW(self, obs, mode):
        if mode == 'v':
            np_img = obs['v']
            img_tensor = torch.tensor(np_img/255.).permute(2, 0, 1).unsqueeze(0).float().to("cuda:0")
            vision = self.model['VAE'].encode({'img': img_tensor})['z_img'].detach().cpu().numpy()[0]
            return vision
        elif mode == 'attr':
            return obs['attr']
        elif mode == 'gw_attr':
            attr_u = torch.tensor(obs['attr'])
            attr_u = {
                "z_cls": attr_u[:3].to("cuda:0"),
                "z_attr": attr_u[3:].to("cuda:0"),
            }
            attr_gw = self.model['GW'].encode(attr_u, 'attr').detach().cpu().numpy()
            return attr_gw
        elif mode == 'gw_v':
            np_img = obs['v']
            img_tensor = torch.tensor(np_img/255.).permute(2, 0, 1).unsqueeze(0).float().to("cuda:0")
            img = self.model['VAE'].encode({'img': img_tensor})['z_img'].detach()[0]
            img_gw = self.model['GW'].encode({'z_img': img}, 'v').detach().cpu().numpy()
            return img_gw
        

class NormWrapper(ObservationWrapper):
    def __init__(self, env, norm):
        super().__init__(env)
        self.norm = norm

    def observation(self, obs):
        return self._norm(obs, self.norm)
     
    def _norm(self, obs, norm):
        if norm == None:
            return obs
        else:
            return (obs - np.array(norm['mean'])) / np.array(norm['std'])