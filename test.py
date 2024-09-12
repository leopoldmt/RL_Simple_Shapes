from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from gymnasium.wrappers import TimeLimit, FrameStack
from Simple_Shapes_RL.utils import NRepeat

import torch
import numpy as np
import os
import yaml
import re

from bim_gw.modules.domain_modules import VAE
from bim_gw.modules.domain_modules.simple_shapes import SimpleShapesAttributes
from bim_gw.modules import GlobalWorkspace

from Simple_Shapes_RL.Env import Simple_Env
from Simple_Shapes_RL.gw_wrapper import GWWrapper, NormWrapper


policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                     net_arch=[dict(pi=[128, 128, 128], vf=[128, 128, 128])])

MODE = {'attr': ['attr'],
        'v': ['v'],
        'gw_attr': ['gw_attr', 'gw_v'],
        'gw_v': ['gw_v', 'gw_attr']
        }

MODE_PATH = {'attr': 'attr', 'v': 'v', 'gw_attr': 'CLIPattr', 'gw_v': 'CLIPv'}

current_directory = os.getcwd()
os.environ["SS_PATH"] = os.getcwd()

path_matcher = re.compile(r'\$\{([^}^{]+)\}')
scientific_number_matcher = re.compile(u'''^(?:
                                        [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                                        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
                                        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
                                        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
                                        |[-+]?\\.(?:inf|Inf|INF)
                                        |\\.(?:nan|NaN|NAN))$''', re.X
                                    )

def path_constructor(loader, node):
  value = node.value
  match = path_matcher.match(value)
  env_var = match.group()[2:-1]
  return os.environ.get(env_var) + value[match.end():]

yaml.add_implicit_resolver('!path', path_matcher)
yaml.add_implicit_resolver(u'tag:yaml.org,2002:float', scientific_number_matcher, list(u'-+0123456789.'))
yaml.add_constructor('!path', path_constructor)
    
if __name__ == '__main__':

    with open('cfg/cfg_test.yaml', encoding="utf-8") as f:
        config = yaml.full_load(f)

    vae = VAE.load_from_checkpoint(config['models_path']['VAE'], strict=False).eval().to("cuda:0")
    domains = {'v': vae.eval(), 'attr': SimpleShapesAttributes(32).eval()}
    gw = GlobalWorkspace.load_from_checkpoint(config['models_path']['GW'], domain_mods=domains, strict=False).eval().to("cuda:0")
    gw_model = {'VAE': vae, 'GW': gw}

    for mode in MODE[config['mode']]:
        env = Simple_Env(render_mode=None)
        env = GWWrapper(env, model=gw_model, mode=mode)
        env = NormWrapper(env, norm=config['normalize'])
        env = TimeLimit(env, max_episode_steps=config['episode_len'])
        env = NRepeat(env, num_frames=config['n_repeats'])
        env = FrameStack(env, 4)
        env = Monitor(env, allow_early_resets=True)
        env = DummyVecEnv([lambda: env])

        model = PPO.load(f"/home/leopold/Documents/Projets/RL/RL_Simple_Shapes/models/CLIP/{config['checkpoint']}/model")

        obs = env.reset()

        i = 0
        len_episode = 0
        total_reward = 0
        reward_array = np.zeros(1000)
        len_array = np.zeros(1000)
        while i != 1000:
            action, _states = model.predict(obs[0])  # VecEnv --> list Env if more than one
            obs, reward, done, info = env.step(np.array([action]))  # VecEnv --> list Env if more than one
            len_episode += 1
            total_reward += reward
            if done:
                reward_array[i] = total_reward
                len_array[i] = len_episode
                i += 1
                len_episode = 0
                total_reward = 0
                obs = env.reset()
        
        if not os.path.exists(current_directory + f"/results/inference/{MODE_PATH[config['mode']]}/"): 
            os.makedirs(current_directory + f"/results/inference/{MODE_PATH[config['mode']]}/") 

        np.save(current_directory + f"/results/inference/{MODE_PATH[config['mode']]}/reward_{MODE_PATH[mode]}_from_{MODE_PATH[config['mode']]}_{config['checkpoint']}", reward_array)
        np.save(current_directory + f"/results/inference/{MODE_PATH[config['mode']]}/len_{MODE_PATH[mode]}_from_{MODE_PATH[config['mode']]}_{config['checkpoint']}", len_array)
