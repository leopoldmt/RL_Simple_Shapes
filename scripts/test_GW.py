from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from gymnasium.wrappers import TimeLimit, FrameStack
from Simple_Shapes_RL.utils import NRepeat

import numpy as np
import os

from Simple_Shapes_RL.Env import Simple_Env
from bim_gw.modules.domain_modules.simple_shapes import SimpleShapesAttributes
from bim_gw.modules import GlobalWorkspace
from bim_gw.modules.domain_modules import VAE


CONFIG = {
    "mode": "GW_attributes",
    "model": "PPO",
    "task": "position_rotation",
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

current_directory = os.path.abspath(os.path.join(os.getcwd(), os.pardir))


if __name__ == '__main__':

    models_path = {'VAE': f'{current_directory}/Simple_Shapes_RL/822888/epoch=282-step=1105680.ckpt', 'GW': f'{current_directory}/Simple_Shapes_RL/GW_cont_gvvjei42/checkpoints/epoch=97-step=191492.ckpt'}
    # models_path = {'VAE': f'{current_directory}/Simple_Shapes_RL/822888/epoch=282-step=1105680.ckpt', 'GW': f'{current_directory}/Simple_Shapes_RL/xbyve6cr/checkpoints/epoch=96-step=189538.ckpt'}

    env = Simple_Env(render_mode=None, task=CONFIG['task'], obs_mode=CONFIG['mode'], model_path=models_path)
    env = TimeLimit(env, max_episode_steps=CONFIG['episode_len'])
    env = NRepeat(env, num_frames=CONFIG['n_repeats'])
    env = FrameStack(env, 2)
    env = Monitor(env, allow_early_resets=True)
    env = DummyVecEnv([lambda: env])

    model = PPO.load(
        f"/home/leopold/Documents/Projets/Arena/RL/Simple_Shapes/Simple_Shapes_RL/models/{CONFIG['checkpoint']}/model")

    obs = env.reset()

    obs_array = np.zeros((50000, 12))
    for i in range(50000):
        action, _states = model.predict(obs[0])  # VecEnv --> list Env if more than one
        obs, reward, done, info = env.step(np.array([action]))  # VecEnv --> list Env if more than one
        obs_array[i, :] = obs[0][0]
        if done:
            obs = env.reset()

    print(np.max(obs_array))
    print(np.min(obs_array))

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(6, 2)
    for i in range(12):
        ax[i // 2, i % 2].hist(obs_array[:, i], 75)
        print(i, np.mean(obs_array[:, i]), np.std(obs_array[:, i]))
    fig.tight_layout()
    plt.show()

    fig, ax = plt.subplots(6, 2)
    for i in range(12):
        ax[i // 2, i % 2].hist((obs_array[:, i] - np.mean(obs_array[:, i])) / np.std(obs_array[:, i]) * np.sqrt(0.01), 75)
    fig.tight_layout()
    plt.show()

    fig, ax = plt.subplots(6, 2)
    for i in range(12):
        ax[i // 2, i % 2].hist(((obs_array[:, i] - np.min(obs_array[:, i])) / (np.max(obs_array[:, i]) - np.min(obs_array[:, i]))) * 2 - 1, 75)
    fig.tight_layout()
    plt.show()

    print('done')


