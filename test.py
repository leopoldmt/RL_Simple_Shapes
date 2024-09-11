from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from gymnasium.wrappers import TimeLimit, FrameStack
from Simple_Shapes_RL.utils import NRepeat

import torch
import numpy as np
import os

from bim_gw.modules.domain_modules import VAE
from bim_gw.modules.domain_modules.simple_shapes import SimpleShapesAttributes
from bim_gw.modules import GlobalWorkspace

from Simple_Shapes_RL.Env import Simple_Env
from Simple_Shapes_RL.gw_wrapper import GWWrapper


policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                     net_arch=[dict(pi=[128, 128, 128], vf=[128, 128, 128])])


NORM_GW_CONT = {'mean': np.array([0.011346655145489494,
                                  0.018906326077828415,
                                  0.0076558825294987766,
                                  0.01374020448433468,
                                  -0.0017686202570563181,
                                  -0.0017319813413190423,
                                  0.018027108799475826,
                                  -0.0032386721732508158,
                                  0.0118084945299651,
                                  0.0192551973626361,
                                  0.010257933979583323,
                                  0.006335173079147935]),
                'std': np.array([0.0564022526881355,
                                 0.10165699320007751,
                                 0.05140064675555506,
                                 0.07199728651151278,
                                 0.05974908398748707,
                                 0.058614378146887865,
                                 0.1330258990665705,
                                 0.053615264789259556,
                                 0.0524300244234034,
                                 0.08749223974877422,
                                 0.05585023656551657,
                                 0.09937382650123902])
                }


CONFIG = {
    "mode": "gw_v",
    "model": "PPO",
    "normalize": NORM_GW_CONT,
    "task": "position_rotation",
    "total_timesteps": 1e6,
    "shape": "(16,16)",
    "target": "fixed",
    "episode_len": 100,
    'n_repeats': 1,
    'checkpoint': 'ht3ebgdl'
}

MODE = {'attr': ['attr'],
        'v': ['v'],
        'gw_attr': ['gw_attr', 'gw_v'],
        'gw_v': ['gw_v', 'gw_attr']
        }

MODE_PATH = {'attr': 'attr', 'v': 'v', 'gw_attr': 'CLIPattr', 'gw_v': 'CLIPv'}

current_directory = os.getcwd()
    
if __name__ == '__main__':

    # models_path = {'VAE': f'{current_directory}/Simple_Shapes_RL/822888/epoch=282-step=1105680.ckpt', 'GW': f'{current_directory}/Simple_Shapes_RL/xbyve6cr/checkpoints/epoch=96-step=189538.ckpt'}
    models_path = {'VAE': f'{current_directory}/Simple_Shapes_RL/checkpoints/VAE_ckpt/epoch=282-step=1105680.ckpt',
                   'GW': f'{current_directory}/Simple_Shapes_RL/checkpoints/GW_cont_ckpt/checkpoints/epoch=97-step=191492.ckpt'}
    # models_path = {'VAE': f'{current_directory}/Simple_Shapes_RL/822888/epoch=282-step=1105680.ckpt',
    #                'GW': f'{current_directory}/Simple_Shapes_RL/GW_trad_cont_xmkeoe5m/checkpoints/epoch=99-step=195400.ckpt'}

    vae = VAE.load_from_checkpoint(models_path['VAE'], strict=False).eval().to("cuda:0")
    domains = {'v': vae.eval(), 'attr': SimpleShapesAttributes(32).eval()}
    gw = GlobalWorkspace.load_from_checkpoint(models_path['GW'], domain_mods=domains, strict=False).eval().to("cuda:0")
    gw_model = {'VAE': vae, 'GW': gw}

    for mode in MODE[CONFIG['mode']]:
        env = Simple_Env(render_mode=None)
        env = GWWrapper(env, model=gw_model, mode=mode)
        env = TimeLimit(env, max_episode_steps=CONFIG['episode_len'])
        env = NRepeat(env, num_frames=CONFIG['n_repeats'])
        env = FrameStack(env, 4)
        env = Monitor(env, allow_early_resets=True)
        env = DummyVecEnv([lambda: env])

        model = PPO.load(f"/home/leopold/Documents/Projets/RL/RL_Simple_Shapes/models/CLIP/{CONFIG['checkpoint']}/model")

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
        
        if not os.path.exists(current_directory + f"/results/inference/{MODE_PATH[CONFIG['mode']]}/"): 
            os.makedirs(current_directory + f"/results/inference/{MODE_PATH[CONFIG['mode']]}/") 

        np.save(current_directory + f"/results/inference/{MODE_PATH[CONFIG['mode']]}/reward_{MODE_PATH[mode]}_from_{MODE_PATH[CONFIG['mode']]}_{CONFIG['checkpoint']}", reward_array)
        np.save(current_directory + f"/results/inference/{MODE_PATH[CONFIG['mode']]}/len_{MODE_PATH[mode]}_from_{MODE_PATH[CONFIG['mode']]}_{CONFIG['checkpoint']}", len_array)
