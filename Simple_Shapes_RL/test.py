from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from gymnasium.wrappers import TimeLimit, FrameStack
from utils import NRepeat

import torch
import numpy as np
import os

from Env import Simple_Env


policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                     net_arch=[dict(pi=[128, 128, 128], vf=[128, 128, 128])])
CONFIG = {
    "mode": "GW_vision",
    "model": "PPO",
    "total_timesteps": 1e6,
    "shape": "(16,16)",
    "target": "fixed",
    "episode_len": 100,
    'n_repeats': 1,
    'checkpoint': 'z9okqbwc'
}

MODE = {'attributes': ['attributes'],
        'vision': ['vision'],
        'GW_attributes': ['GW_attributes', 'GW_vision'],
        'GW_vision': ['GW_vision', 'GW_attributes']
        }

MODE_PATH = {'attributes': 'attr', 'vision': 'v', 'GW_attributes': 'GWattr', 'GW_vision': 'GWv'}

current_directory = os.getcwd()


if __name__ == '__main__':

    models_path = {'VAE': '822888/epoch=282-step=1105680.ckpt', 'GW': 'xbyve6cr/checkpoints/epoch=96-step=189538.ckpt'}

    for mode in MODE[CONFIG['mode']]:
        env = Simple_Env(render_mode=None, task='position_rotation', obs_mode=mode, model_path=models_path)
        env = TimeLimit(env, max_episode_steps=CONFIG['episode_len'])
        env = NRepeat(env, num_frames=CONFIG['n_repeats'])
        env = FrameStack(env, 2)
        env = Monitor(env, allow_early_resets=True)
        env = DummyVecEnv([lambda: env])

        model = PPO.load(f"/home/leopold/Documents/Projets/Arena/RL/Simple_Shapes/models/{CONFIG['checkpoint']}/model")

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