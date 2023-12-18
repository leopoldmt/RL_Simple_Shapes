from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from gymnasium.wrappers import TimeLimit, FrameStack
from Simple_Shapes_RL.utils import NRepeat

import torch
import numpy as np
import os

from Simple_Shapes_RL.Env import Simple_Env


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
    "mode": "GW_vision",
    "model": "PPO",
    "normalize": None,
    "task": "position_rotation",
    "total_timesteps": 1e6,
    "shape": "(16,16)",
    "target": "fixed",
    "episode_len": 100,
    'n_repeats': 1,
    'checkpoint': '9zixbcbh'
}

MODE = {'attributes': ['attributes'],
        'vision': ['vision'],
        'GW_attributes': ['GW_attributes', 'GW_vision'],
        'GW_vision': ['GW_vision', 'GW_attributes']
        }

MODE_PATH = {'attributes': 'attr', 'vision': 'v', 'GW_attributes': 'CLIPattr', 'GW_vision': 'CLIPv'}

current_directory = os.getcwd()


if __name__ == '__main__':

    # models_path = {'VAE': f'{current_directory}/Simple_Shapes_RL/822888/epoch=282-step=1105680.ckpt', 'GW': f'{current_directory}/Simple_Shapes_RL/xbyve6cr/checkpoints/epoch=96-step=189538.ckpt'}
    models_path = {'VAE': f'{current_directory}/Simple_Shapes_RL/822888/epoch=282-step=1105680.ckpt',
                   'GW': f'{current_directory}/Simple_Shapes_RL/GW_cont_gvvjei42/checkpoints/epoch=97-step=191492.ckpt'}
    # models_path = {'VAE': f'{current_directory}/Simple_Shapes_RL/822888/epoch=282-step=1105680.ckpt',
    #                'GW': f'{current_directory}/Simple_Shapes_RL/GW_trad_cont_xbyve6cr/checkpoints/epoch=96-step=189538.ckpt'}

    for mode in MODE[CONFIG['mode']]:
        env = Simple_Env(render_mode=None, task=CONFIG['task'], obs_mode=mode, model_path=models_path, normalize=CONFIG['normalize'])
        env = TimeLimit(env, max_episode_steps=CONFIG['episode_len'])
        env = NRepeat(env, num_frames=CONFIG['n_repeats'])
        env = FrameStack(env, 4)
        env = Monitor(env, allow_early_resets=True)
        env = DummyVecEnv([lambda: env])

        model = PPO.load(f"/home/leopold/Documents/Projets/Arena/RL/Simple_Shapes/Simple_Shapes_RL/models/{CONFIG['checkpoint']}/model")

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

        np.save(current_directory + f"/results/inference/{MODE_PATH[CONFIG['mode']]}/reward_{MODE_PATH[mode]}_from_{MODE_PATH[CONFIG['mode']]}_{CONFIG['checkpoint']}", reward_array)
        np.save(current_directory + f"/results/inference/{MODE_PATH[CONFIG['mode']]}/len_{MODE_PATH[mode]}_from_{MODE_PATH[CONFIG['mode']]}_{CONFIG['checkpoint']}", len_array)