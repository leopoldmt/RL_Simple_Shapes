from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from gymnasium.wrappers import TimeLimit, FrameStack
from Simple_Shapes_RL.utils import NRepeat

import torch
import numpy as np
import wandb

from Simple_Shapes_RL.Env import Simple_Env

from bim_gw.modules.domain_modules import VAE
from bim_gw.modules.domain_modules.simple_shapes import SimpleShapesAttributes
from bim_gw.modules import GlobalWorkspace


policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                     net_arch=[dict(pi=[128, 128, 128], vf=[128, 128, 128])])
CONFIG = {
    "mode": "GW_attributes",
    "model": "PPO",
    "total_timesteps": 1e6,
    "shape": "(16,16)",
    "target": "fixed",
    "episode_len": 100,
    'n_repeats': 1
}


if __name__ == '__main__':

    vae = VAE.load_from_checkpoint(
        'Simple_Shapes_RL/822888/epoch=282-step=1105680.ckpt',
        strict=False,
    )

    domains = {'v': vae.eval(), 'attr': SimpleShapesAttributes(32).eval()}

    gw = GlobalWorkspace.load_from_checkpoint('Simple_Shapes_RL/xbyve6cr/checkpoints/epoch=96-step=189538.ckpt', domain_mods=domains,
                                              strict=False).eval()

    models = {'VAE': vae, 'GW': gw}

    env = Simple_Env(render_mode='image', task='position_rotation', obs_mode=CONFIG['mode'], model=models)
    env = TimeLimit(env, max_episode_steps=CONFIG['episode_len'])
    env = NRepeat(env, num_frames=CONFIG['n_repeats'])
    env = FrameStack(env, 2)
    env = Monitor(env, allow_early_resets=True)
    env = DummyVecEnv([lambda: env])

    model = PPO.load('/home/leopold/Documents/Projets/Arena/RL/Simple_Shapes/models/yv294erb/model')

    obs = env.reset()
    for i in range(10000):
        action, _states = model.predict(obs[0])  # VecEnv --> list Env if more than one
        obs, reward, done, info = env.step(np.array([action]))  # VecEnv --> list Env if more than one
        if done:
            obs = env.reset()