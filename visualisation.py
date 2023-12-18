from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from gymnasium.wrappers import TimeLimit, FrameStack
from Simple_Shapes_RL.utils import NRepeat

import torch
import numpy as np
import wandb
import os

from Simple_Shapes_RL.Env import Simple_Env

from bim_gw.modules.domain_modules import VAE
from bim_gw.modules.domain_modules.simple_shapes import SimpleShapesAttributes
from bim_gw.modules import GlobalWorkspace


policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                     net_arch=[dict(pi=[128, 128, 128], vf=[128, 128, 128])])
CONFIG = {
    "mode": "GW_attributes",
    "model": "PPO",
    "task": "position_rotation",
    "total_timesteps": 1e6,
    "target": "random_GW_attributes",
    "shape": "(16,16)",
    "episode_len": 75,
    'n_repeats': 1,
    'checkpoint': 'kjor1qs1'
}

current_directory = os.getcwd()


if __name__ == '__main__':

    models_path = {'VAE': f'{current_directory}/Simple_Shapes_RL/822888/epoch=282-step=1105680.ckpt', 'GW': f'{current_directory}/Simple_Shapes_RL/xbyve6cr/checkpoints/epoch=96-step=189538.ckpt'}
    # models_path = {'VAE': f'{current_directory}/Simple_Shapes_RL/822888/epoch=282-step=1105680.ckpt',
    #                'GW': f'{current_directory}/Simple_Shapes_RL/GW_cont_gvvjei42/checkpoints/epoch=97-step=191492.ckpt'}

    env = Simple_Env(render_mode='image', task=CONFIG['task'], obs_mode=CONFIG['mode'], model_path=models_path, target_mode=CONFIG['target'])
    env = TimeLimit(env, max_episode_steps=CONFIG['episode_len'])
    env = NRepeat(env, num_frames=CONFIG['n_repeats'])
    env = FrameStack(env, 2)
    env = Monitor(env, allow_early_resets=True)
    env = DummyVecEnv([lambda: env])

    model = PPO.load(
        f"/home/leopold/Documents/Projets/Arena/RL/Simple_Shapes/models/{CONFIG['checkpoint']}/model")

    obs = env.reset()
    for i in range(10000):
        action, _states = model.predict(obs[0])  # VecEnv --> list Env if more than one
        obs, reward, done, info = env.step(np.array([action]))  # VecEnv --> list Env if more than one
        if done:
            obs = env.reset()