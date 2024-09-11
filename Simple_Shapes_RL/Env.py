from typing import Any
import numpy as np
import pygame
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.backends.backend_agg as agg
# matplotlib.use("Agg")
from typing import cast
import pylab
from PIL import Image
import torchvision.transforms as transforms
import torch
from torch.nn import functional as F

import gymnasium as gym

from Simple_Shapes_RL.utils import get_obs_space, get_action_space, generate_new_attributes, generate_image


class Simple_Env(gym.Env):

    def __init__(self, render_mode=None, obs_mode='dict'):

        self.obs_mode = obs_mode
        self.render_mode = render_mode
        self.target = np.array([16, 16, 0])

        self.reward_function = None

        self.window = None
        self.clock = None

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
        return get_obs_space(self.obs_mode)

    @property
    def action_space(self):
        return get_action_space()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        #initialize attributes randomly
        self.attributes = generate_new_attributes()

        self.init_dist = np.linalg.norm(self.target[:2] - self.attributes[1:3])
        self.init_angle = np.abs(self.target[2] - self.attributes[4]) % np.pi

        observation = self._get_obs()
        info = self._get_info()

        self._render_frame()

        return observation, info

    def step(self, action):

        self.attributes[1] += self._action_to_direction[action][0]
        self.attributes[2] += self._action_to_direction[action][1]
        self.attributes[4] = (self.attributes[4] + self._action_to_direction[action][2]) % (2*np.pi)

        observation = self._get_obs()
        info = self._get_info()

        # compute reward
        reward = self._get_task_reward()

        # compute terminated
        terminated = self._get_task_terminated()

        self._render_frame()

        return observation, reward, terminated, False, info

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def render(self):
        if self.render_mode == "rgb":
            return self._render_frame()

    def _transform_action(self, action):
        x = action[:2]
        action[:2] = (x + (2/3)*(x > 0) - (2/3)*(x < 0)).astype(int)
        return action

    def _render_frame(self):
        if self.render_mode == "rgb":
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

            if self.window is None and self.render_mode == "rgb":
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

            if self.clock is None and self.render_mode == "rgb":
                self.clock = pygame.time.Clock()

    def _get_obs(self):
        if self.obs_mode == 'dict':
            return {'attr': self._get_attributes(), 'v': self._get_vision()}
        elif self.obs_mode == 'attributes':
            return self._get_attributes()
        elif self.obs_mode == 'vision':
            return self._get_vision()
            
    def _get_attributes(self):
        attr = self.attributes[:8]

        attr_enc = torch.zeros(8)
        attr_enc[0:2] = torch.tensor(attr[1:3] / 32)
        attr_enc[2] = torch.tensor((attr[3] - 7) / 14)
        attr_enc[3] = (np.cos(attr[4]) + 1) / 2
        attr_enc[4] = (np.sin(attr[4]) + 1) / 2
        attr_enc[5:8] = torch.tensor(attr[5:8] / 255.)
        attr_enc = attr_enc * 2 - 1
        attr_enc = torch.cat((F.one_hot(torch.tensor(attr[0]).to(torch.int64), 3).type_as(attr_enc), attr_enc), dim=0)
        return attr_enc.numpy()

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

        return np_img

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self.attributes[1:3] - self.target[:2]
            )
        }

    def _get_task_reward(self):
        reward_pos = -np.linalg.norm(self.attributes[1:3] - self.target[:2])
        angle_vec = np.array([np.cos(self.attributes[4]), np.sin(self.attributes[4])])
        target_vec = np.array([np.cos(self.target[2]), np.sin(self.target[2])])
        reward_rot = -np.abs(np.arccos(np.clip(np.dot(angle_vec, target_vec), -1.0, 1.0)))
        return reward_pos + 10 * reward_rot

    def _get_task_terminated(self):
        terminated_pos = np.linalg.norm(self.attributes[1:3] - self.target[:2], ord=1) == 0.
        angle_vec = np.array([np.cos(self.attributes[4]), np.sin(self.attributes[4])])
        target_vec = np.array([np.cos(self.target[2]), np.sin(self.target[2])])
        terminated_rot = np.arccos(np.clip(np.dot(angle_vec, target_vec), -1.0, 1.0)) < np.pi/32
        terminated = terminated_pos and terminated_rot
        return terminated


if __name__=='__main__':

    env = Simple_Env(render_mode=None, obs_mode='dict')
    obs = env.reset()
    for i in range(100):
        observation, reward, terminated, truncated, info = env.step(1)
        print(reward)
        if terminated:
            print('reset')
            env.reset()
    env.close()
