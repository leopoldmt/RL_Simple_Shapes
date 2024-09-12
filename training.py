from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from gymnasium.wrappers import TimeLimit, FrameStack
from Simple_Shapes_RL.utils import NRepeat
import os
import yaml
import torch
import numpy as np
import wandb
import re
from wandb.integration.sb3 import WandbCallback

from bim_gw.modules.domain_modules import VAE
from bim_gw.modules.domain_modules.simple_shapes import SimpleShapesAttributes
from bim_gw.modules import GlobalWorkspace

from Simple_Shapes_RL.Env import Simple_Env
from Simple_Shapes_RL.gw_wrapper import GWWrapper

policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                     net_arch=[dict(pi=[64, 64, 64], vf=[128, 128, 128])])

current_directory = os.getcwd()
os.environ["SS_PATH"] = os.getcwd()

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


def make_env(rank, seed = 0, model=None, config=None, monitor_dir=None, wrapper_class=None, monitor_kwargs=None, wrapper_kwargs=None):
    def _init():
        env = Simple_Env(render_mode=None)
        env = GWWrapper(env, model=model, mode=config['mode'])
        env = TimeLimit(env, max_episode_steps=config['episode_len'])
        env = NRepeat(env, num_frames=config['n_repeats'])
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

    # run = wandb.init(
    #     project="RL_factory",
    #     config=CONFIG,
    #     sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    # )

    with open('cfg.yaml', encoding="utf-8") as f:
        config = yaml.full_load(f)

    seed = np.random.randint(0, 1000)
    vae = VAE.load_from_checkpoint(config['models_path']['VAE'], strict=False).eval().to("cuda:0")
    domains = {'v': vae.eval(), 'attr': SimpleShapesAttributes(32).eval()}
    gw = GlobalWorkspace.load_from_checkpoint(config['models_path']['GW'], domain_mods=domains, strict=False).eval().to("cuda:0")
    gw_model = {'VAE': vae, 'GW': gw}
    env = DummyVecEnv([make_env(i, seed=seed, model=gw_model, config=config) for i in range(config['n_envs'])])

    model = PPO('MlpPolicy',
                env,
                learning_rate=config['lr'],
                n_steps=int(config['n_steps'] / config['n_envs']),
                batch_size=config['mini_batch_size'],
                n_epochs=config['num_epochs'],
                gamma=config['gamma'],
                gae_lambda=config['gae_lambda'],
                ent_coef=config['ent_coef'],
                vf_coef=config['vf_coef'],
                policy_kwargs=config['policy_kwargs'],
                # learning_starts=5000,
                # buffer_size=50000,
                verbose=1,
                # tensorboard_log=f"runs/{run.id}"
    )

    model.learn(total_timesteps=config['total_timesteps'],
        progress_bar=True,
        # callback=WandbCallback(
        #     model_save_freq=100,
        #     model_save_path=f"models/{run.id}",
        # )
    )

    # run.finish()

