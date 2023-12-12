from typing import Any
import numpy as np
import pygame
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.backends.backend_agg as agg
# matplotlib.use("Agg")
import io
import time
from typing import cast
import pylab
from PIL import Image
import torchvision.transforms as transforms
import torch
from torch.nn import functional as F

import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ObsType, RenderFrame

from Simple_Shapes_RL.utils import get_obs_space, get_action_space, generate_new_attributes, generate_new_target, generate_image
from bim_gw.modules.domain_modules import VAE
from bim_gw.modules.domain_modules.simple_shapes import SimpleShapesAttributes
from bim_gw.modules import GlobalWorkspace


class Simple_Env(gym.Env):

    def __init__(self, render_mode=None, task='position', obs_mode='attributes', target_mode='fixed', model_path=None, normalize=None):

        self.obs_mode = obs_mode.split('_')
        self.model = {'VAE': None, 'GW': None}

        if ('vision' in self.obs_mode[0]) or ('GW' in self.obs_mode[0]):
            vae = VAE.load_from_checkpoint(
                model_path['VAE'],
                strict=False,
            )
            self.model['VAE'] = vae.eval().to("cuda:0")

        if 'GW' in self.obs_mode[0]:
            domains = {'v': vae.eval(), 'attr': SimpleShapesAttributes(32).eval()}
            gw = GlobalWorkspace.load_from_checkpoint(model_path['GW'], domain_mods=domains,
                                                  strict=False).eval()
            self.model['GW'] = gw.eval().to("cuda:0")

        self.render_mode = render_mode
        self.target_mode = target_mode.split('_')
        # self.model = model

        self.task = task
        self.reward_function = None

        self.window = None
        self.clock = None
        self.normalize = normalize

        self._action_to_direction = {
            0: np.array([0, 0, 0]),
            1: np.array([1, 0, 0]),
            2: np.array([0, 1, 0]),
            3: np.array([-1, 0, 0]),
            4: np.array([0, -1, 0]),
            5: np.array([0, 0, np.pi/32]),
            6: np.array([0, 0, -np.pi/32]),
        }

    @property
    def observation_space(self):
        return get_obs_space(self.obs_mode[0], self.target_mode)

    @property
    def action_space(self):
        return get_action_space(self.task)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        #initialize attributes randomly
        self.attributes = generate_new_attributes(self.target_mode[0])

        #initialize target
        self.target = generate_new_target(self.target_mode[0])
        if self.target_mode[0] == 'random':
            self.target_modality = self._get_target_modality()

        self.init_dist = np.linalg.norm(self.target[:2] - self.attributes[1:3])
        self.init_angle = np.abs(self.target[2] - self.attributes[4]) % np.pi

        observation = self._get_obs()
        info = self._get_info()

        self._render_frame()

        # print(self.attributes)

        return observation, info

    def step(self, action):

        # action = action * self._act_high
        # action = self._transform_action(action)
        #
        # #compute new position
        # self.attributes[1] += action[0]
        # self.attributes[2] += action[1]
        #
        # #compute new rotation
        # self.attributes[4] = (self.attributes[4] + action[2])%(2*np.pi)
        self.attributes[1] += self._action_to_direction[action][0]
        self.attributes[2] += self._action_to_direction[action][1]
        self.attributes[4] = (self.attributes[4] + self._action_to_direction[action][2]) % (2*np.pi)

        # print(self.attributes)

        #compute reward
        reward = self._get_task_reward(self.task)

        #compute terminated
        terminated, reward = self._get_task_terminated(self.task, reward)

        observation = self._get_obs()
        info = self._get_info()

        # margin = 14 // 2
        # if self.attributes[1] < margin or self.attributes[1] > 32 - margin or self.attributes[2] < margin or self.attributes[2] > 32 - margin:
        #     terminated = True
        #     reward -= 1000

        self._render_frame()

        # print(self.attributes)
        # print(reward)

        return observation, reward, terminated, False, info

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def render(self):
        if self.render_mode == "image":
            return self._render_frame()

    def _transform_action(self, action):
        x = action[:2]
        action[:2] = (x + (2/3)*(x > 0) - (2/3)*(x < 0)).astype(int)
        return action

    def _render_frame(self):
        if self.render_mode == "image":
            dpi = 1
            imsize = 32
            fig = pylab.figure(figsize=[4, 4],  # Inches
                               dpi=100,  # 100 dots per inch, so the resulting buffer is 400x400 pixels
                               )
            ax = fig.gca()
            ax = cast(plt.Axes, ax)
            generate_image(ax, self.attributes[0], self.attributes[1:3], self.attributes[3], self.attributes[4],
                           self.attributes[5:8], imsize)
            ax.set_facecolor("black")

            if self.target_mode[0] == 'random':
                ax.plot(self.target[0], self.target[1], '+', color='red')
                plt.arrow(self.target[0], self.target[1], -np.sin(self.target[2])*5, np.cos(self.target[2])*5,
                          head_width=1.,
                          width=0.05,
                          color='red')

            canvas = agg.FigureCanvasAgg(fig)
            canvas.draw()
            plt.close(fig)
            renderer = canvas.get_renderer()
            raw_data = renderer.tostring_rgb()

            if self.window is None and self.render_mode == "image":
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode(
                    (400, 400)
                )

            screen = pygame.display.get_surface()

            size = canvas.get_width_height()
            surf = pygame.image.fromstring(raw_data, size, "RGB")
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            if self.clock is None and self.render_mode == "image":
                self.clock = pygame.time.Clock()

    def _get_target_modality(self):
        target = self._get_attributes_target()
        if self.target_mode[1] == 'vision':
            return self._get_vision('target')
        elif self.target_mode[1] == 'GW':
            return self._get_GW(self.target_mode[2], mode='target')
        return target

    def _get_obs(self):
        if self.target_mode[0] == 'fixed':
            if self.obs_mode[0] == 'attributes':
                return self._get_attributes()
            elif self.obs_mode[0] == 'vision':
                return self._get_vision()
            elif self.obs_mode[0] == 'GW':
                return self._get_GW(self.obs_mode[1])
        elif self.target_mode[0] == 'random':
            if self.obs_mode[0] == 'attributes':
                return np.concatenate((self._get_attributes(), self.target_modality))
            elif self.obs_mode[0] == 'vision':
                return np.concatenate((self._get_vision(), self.target_modality))
            elif self.obs_mode[0] == 'GW':
                return np.concatenate((self._get_GW(self.obs_mode[1]), self.target_modality))

    def _get_attributes(self):
        attr = self.attributes[:8]

        attr_enc = torch.zeros(8)
        attr_enc[0:2] = torch.tensor(attr[1:3] / 32)
        attr_enc[2] = torch.tensor(attr[3] / 14)
        attr_enc[3] = (np.cos(attr[4]) + 1) / 2
        attr_enc[4] = (np.sin(attr[4]) + 1) / 2
        attr_enc[5:8] = torch.tensor(attr[5:8] / 255.)
        attr_enc = attr_enc * 2 - 1
        attr_enc = torch.cat((F.one_hot(torch.tensor(attr[0]).to(torch.int64), 3).type_as(attr_enc), attr_enc), dim=0)
        return attr_enc.numpy()

        # return np.array([attr[0], attr[1], attr[2], attr[3], np.cos(attr[4]), np.sin(attr[4]), attr[5], attr[6], attr[7]])

    def _get_attributes_target(self):
        attr = self.attributes[:8]

        attr_enc = torch.zeros(8)
        attr_enc[0] = torch.tensor(self.target[0] / 32)
        attr_enc[1] = torch.tensor(self.target[1] / 32)
        attr_enc[2] = torch.tensor(attr[3] / 14)
        attr_enc[3] = (np.cos(self.target[2]) + 1) / 2
        attr_enc[4] = (np.sin(self.target[2]) + 1) / 2
        attr_enc[5:8] = torch.tensor(attr[5:8] / 255.)
        attr_enc = attr_enc * 2 - 1
        attr_enc = torch.cat((F.one_hot(torch.tensor(attr[0]).to(torch.int64), 3).type_as(attr_enc), attr_enc), dim=0)
        return attr_enc.numpy()

        # return np.array([attr[0], self.target[0], self.target[1], attr[3], np.cos(self.target[2]), np.sin(self.target[2]), attr[5], attr[6], attr[7]])

    def _get_vision(self, mode='state'):
        fig, ax = plt.subplots(figsize=(32, 32), dpi=1)
        ax = cast(plt.Axes, ax)

        if mode == 'state':
            generate_image(ax, self.attributes[0], self.attributes[1:3], self.attributes[3], self.attributes[4],
                           self.attributes[5:8], 32)

        elif mode == 'target':
            generate_image(ax, self.attributes[0], self.target[:2], self.attributes[3], self.target[2], self.attributes[5:8], 32)

        ax.set_facecolor("black")
        plt.tight_layout(pad=0)

        fig.canvas.draw()
        buf = fig.canvas.tostring_rgb()
        ncols, nrows = fig.canvas.get_width_height()
        plt.close(fig)
        np_img = np.fromstring(buf, dtype=np.uint8).reshape(nrows, ncols, 3)
        img_tensor = torch.tensor(np_img/255.).permute(2, 0, 1).unsqueeze(0).float().to("cuda:0")


        vision = self.model['VAE'].encode({'img': img_tensor})['z_img'].detach().cpu().numpy()[0]

        return vision

    def _get_GW(self, obs_from, mode='state'):
        if obs_from == 'attributes':
            attributes = self.attributes
            if mode == 'target':
                attributes = self._get_attributes_target()

            # encode attributes to unimodal latent
            attr = torch.zeros(8)
            attr[0:3] = torch.tensor(attributes[1:4] / 32)
            attr[3] = (np.cos(attributes[4]) + 1) / 2
            attr[4] = (np.sin(attributes[4]) + 1) / 2
            attr[5:8] = torch.tensor(attributes[5:8] / 255.)
            attr = attr * 2 - 1
            attr_u = {
                "z_cls": F.one_hot(torch.tensor(attributes[0]).to(torch.int64), 3).type_as(attr).to("cuda:0"),
                "z_attr": attr.to("cuda:0"),
            }
            # encode attributes_u to GW
            attr_gw = self.model['GW'].encode(attr_u, 'attr').detach().cpu().numpy()
            if self.normalize is not None:
                attr_gw = (attr_gw - self.normalize['mean']) / self.normalize['std']
            return attr_gw
        elif obs_from == 'vision':
            # encode vision to unimodal latent
            img = self._get_vision(mode)
            # encode vision_u to GW
            img_gw = self.model['GW'].encode({'z_img': torch.tensor(img).to("cuda:0")}, 'v').detach().cpu().numpy()
            if self.normalize is not None:
                img_gw = (img_gw - self.normalize['mean']) / self.normalize['std']
            return img_gw

    def _get_info(self):
        if self.task == 'position':
            return {
                "distance": np.linalg.norm(
                    self.attributes[1:3] - self.target[:2]
                )
            }
        elif self.task == 'rotation':
            return {}
        else:
            return {}

    def _get_task_reward(self, task):
        reward_pos = -np.linalg.norm(self.attributes[1:3] - self.target[:2])
        angle_vec = np.array([np.cos(self.attributes[4]), np.sin(self.attributes[4])])
        target_vec = np.array([np.cos(self.target[2]), np.sin(self.target[2])])
        reward_rot = -np.abs(np.arccos(np.clip(np.dot(angle_vec, target_vec), -1.0, 1.0)))
        # reward_rot = np.abs(self.attributes[4] - self.target[2])
        # reward_rot = -(reward_rot - np.pi * (reward_rot > np.pi))
        return reward_pos + 10 * reward_rot

    def _get_task_terminated(self, task, reward):
        terminated_pos = np.linalg.norm(self.attributes[1:3] - self.target[:2], ord=1) == 0.
        angle_vec = np.array([np.cos(self.attributes[4]), np.sin(self.attributes[4])])
        target_vec = np.array([np.cos(self.target[2]), np.sin(self.target[2])])
        terminated_rot = np.arccos(np.clip(np.dot(angle_vec, target_vec), -1.0, 1.0)) < np.pi/32
        terminated = terminated_pos and terminated_rot
        if task == 'sparse':
            reward = 1 * terminated
        # reward += 1000 * terminated
        return terminated, reward


if __name__=='__main__':
    from bim_gw.modules.domain_modules import VAE

    vae = VAE.load_from_checkpoint(
        '/home/leopold/Documents/Projets/Arena/RL/Simple_Shapes/822888/epoch=282-step=1105680.ckpt',
        strict=False,
    )
    model = {'VAE': vae}
    env = Simple_Env(model=model, render_mode='image', task='position_rotation', obs_mode='attributes')
    obs = env.reset()
    # print(obs)
    for i in range(100):
        observation, reward, terminated, truncated, info = env.step(1)
        print(reward)
        if terminated:
            print('reset')
            env.reset()
        # print(observation, reward, terminated, info)
    env.close()
