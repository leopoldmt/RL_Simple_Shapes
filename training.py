from stable_baselines3 import PPO
from stable_baselines3 import DDPG
from stable_baselines3 import DQN
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from gymnasium.wrappers import TimeLimit, FrameStack
from Simple_Shapes_RL.utils import NRepeat
import os
import torch
import numpy as np
import wandb
from wandb.integration.sb3 import WandbCallback

from Simple_Shapes_RL.Env import Simple_Env

policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                     net_arch=[dict(pi=[64, 64, 64], vf=[128, 128, 128])])

current_directory = os.getcwd()

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
                                 0.09937382650123902] / np.sqrt(0.01))
                }

CONFIG = {
    "models_path": {'VAE': f'{current_directory}/Simple_Shapes_RL/822888/epoch=282-step=1105680.ckpt', 'GW': f'{current_directory}/Simple_Shapes_RL/GW_cont_gvvjei42/checkpoints/epoch=97-step=191492.ckpt'}, # 'GW': '/home/leopold/Documents/Projets/Arena/RL/Simple_Shapes/Simple_Shapes_RL/xbyve6cr/checkpoints/epoch=96-step=189538.ckpt'},
    "mode": "GW_attributes",
    "model": "PPO",
    "task": "position_rotation",
    "normalize": NORM_GW_CONT,
    "total_timesteps": 1e7,
    "shape": "(16,16)",
    "target": "random_GW_attributes",
    "episode_len": 100,
    "n_steps": 16384,
    "num_epochs": 15,
    "mini_batch_size": 512,
    "lr": 3e-4,
    "policy_kwargs": None,
    "gamma": 0.9,
    "gae_lambda": 0.95,
    "target_kl": None,
    "vf_coef": 0.5,
    "ent_coef": 0.,
    'n_repeats': 1,
    'n_envs': 2,
}

def make_env(rank, seed = 0, monitor_dir=None, wrapper_class=None, monitor_kwargs=None, wrapper_kwargs=None):
    def _init():
        env = Simple_Env(render_mode=None, task=CONFIG['task'], obs_mode=CONFIG['mode'], target_mode=CONFIG['target'], model_path=CONFIG['models_path'])
        env = TimeLimit(env, max_episode_steps=CONFIG['episode_len'])
        env = NRepeat(env, num_frames=CONFIG['n_repeats'])
        env = FrameStack(env, 4)

        monitor_path = os.path.join(monitor_dir, str(rank)) if monitor_dir is not None else None
        # Create the monitor folder if needed
        if monitor_path is not None and monitor_dir is not None:
            os.makedirs(monitor_dir, exist_ok=True)
        env = Monitor(env, filename=monitor_path)
        # Optionally, wrap the environment with the provided wrapper
        if wrapper_class is not None:
            env = wrapper_class(env, **wrapper_kwargs)

        env.reset(seed=seed + rank)

        return env

    set_random_seed(seed)
    return _init


if __name__ == '__main__':

    # run = wandb.init(
    #     project="RL_factory",
    #     config=CONFIG,
    #     sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    # )

    seed = np.random.randint(0, 1000)
    env = DummyVecEnv([make_env(i, seed=seed) for i in range(CONFIG['n_envs'])])

    model = PPO('MlpPolicy',
                env,
                learning_rate=CONFIG['lr'],
                n_steps=int(CONFIG['n_steps'] / CONFIG['n_envs']),
                batch_size=CONFIG['mini_batch_size'],
                n_epochs=CONFIG['num_epochs'],
                gamma=CONFIG['gamma'],
                gae_lambda=CONFIG['gae_lambda'],
                ent_coef=CONFIG['ent_coef'],
                vf_coef=CONFIG['vf_coef'],
                policy_kwargs=CONFIG['policy_kwargs'],
                # learning_starts=5000,
                # buffer_size=50000,
                verbose=1,
                # tensorboard_log=f"runs/{run.id}"
    )

    # model = RecurrentPPO("MlpLstmPolicy", env, verbose=1, tensorboard_log=f"runs/{run.id}")

    model.learn(total_timesteps=CONFIG['total_timesteps'],
        progress_bar=True,
        # callback=WandbCallback(
        #     model_save_freq=100,
        #     model_save_path=f"models/{run.id}",
        # )
    )

    # run.finish()

