import numpy as np
from gymnasium import spaces
import matplotlib.path as mpath
from matplotlib import patches as patches
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from typing import NamedTuple, cast

import gymnasium as gym


class NRepeat(gym.Wrapper):
    def __init__(self, env, num_frames):
        super().__init__(env)
        self.num_frames = num_frames
        self.env = env

    def step(self, action):
        total_reward = 0.0

        for x in range(self.num_frames):
            obs, rew, term, trunc, info = super().step(action)
            total_reward += rew
            if term or trunc:
                break

        return obs, total_reward, term, trunc, info

    def reset(self, seed=0):

        obs, info = self.env.reset()

        return obs, info


NB_OBS = {'attributes': 9, 'vision': 12, 'GW': 12}


def get_obs_space(obs_mode):
    if obs_mode == 'dict':
        return spaces.Dict({'attr': spaces.Box(low=np.array([0, -32, -32, 7, -1, -1, 0, 0, 0]), high=np.array([2, 32, 32, 14, 1, 1, 255, 255, 255])),
                            'v': spaces.Box(low=0, high=255, shape=(32,32,3))
                })
    elif obs_mode == 'attributes':
        return spaces.Box(low=-1, high=1, shape=(11,))
    elif obs_mode == 'vision':
        return spaces.Box(low=0, high=255, shape=(32,32,3))


def get_action_space():
    return spaces.Discrete(7)


def get_diamond_patch(
    location: np.ndarray,
    scale: int,
    rotation: float,
    color: np.ndarray,
) -> patches.Polygon:
    x, y = location[0], location[1]
    coordinates = np.array([[0.5, 0.0], [1, 0.3], [0.5, 1], [0, 0.3]])
    origin = np.array([[x, y]])
    patch = patches.Polygon(
        get_transformed_coordinates(coordinates, origin, scale, rotation),
        facecolor=color,
    )
    return patch


def generate_new_target(target_mode):
    if target_mode == 'fixed':
        return np.array([16, 16, 0])
    elif target_mode == 'random':
        position = generate_location(1, 14, 32)[0]
        rotation = generate_rotation(1)
        return np.concatenate((position, rotation), axis=0)


def generate_new_attributes(target_mode=None):
    cls = generate_class(1)
    location = np.array([16., 16.]) if target_mode == 'random' else generate_location(1, 14, 32)[0]
    scale = generate_scale(1, 7, 14)
    rotation = np.array([0.]) if target_mode == 'random' else generate_rotation(1)
    color_rgb, color_hls = generate_color(1, 46, 256)
    return np.concatenate((cls, location, scale, rotation, color_rgb[0], color_hls[0]), axis=0)


def get_transformed_coordinates(
    coordinates: np.ndarray, origin: np.ndarray, scale: float, rotation: float
) -> np.ndarray:
    center = np.array([[0.5, 0.5]])
    rotation_m = np.array(
        [
            [np.cos(rotation), -np.sin(rotation)],
            [np.sin(rotation), np.cos(rotation)],
        ]
    )
    rotated_coordinates = (coordinates - center) @ rotation_m.T
    return origin + scale * rotated_coordinates


def get_triangle_patch(
    location: np.ndarray,
    scale: int,
    rotation: float,
    color: np.ndarray,
) -> patches.Polygon:
    x, y = location[0], location[1]
    origin = np.array([[x, y]])
    coordinates = np.array([[0.5, 1.0], [0.2, 0.0], [0.8, 0.0]])
    patch = patches.Polygon(
        get_transformed_coordinates(coordinates, origin, scale, rotation),
        facecolor=color,
    )
    return patch


def get_egg_patch(
    location: np.ndarray, scale: int, rotation: float, color: np.ndarray
) -> patches.PathPatch:
    x, y = location[0], location[1]
    origin = np.array([[x, y]])
    coordinates = np.array(
        [
            [0.5, 0],
            [0.8, 0],
            [0.9, 0.1],
            [0.9, 0.3],
            [0.9, 0.5],
            [0.7, 1],
            [0.5, 1],
            [0.3, 1],
            [0.1, 0.5],
            [0.1, 0.3],
            [0.1, 0.1],
            [0.2, 0],
            [0.5, 0],
        ]
    )
    codes = [
        mpath.Path.MOVETO,
        mpath.Path.CURVE4,
        mpath.Path.CURVE4,
        mpath.Path.CURVE4,
        mpath.Path.CURVE4,
        mpath.Path.CURVE4,
        mpath.Path.CURVE4,
        mpath.Path.CURVE4,
        mpath.Path.CURVE4,
        mpath.Path.CURVE4,
        mpath.Path.CURVE4,
        mpath.Path.CURVE4,
        mpath.Path.CURVE4,
    ]
    path = mpath.Path(
        get_transformed_coordinates(coordinates, origin, scale, rotation),
        codes,
    )
    patch = patches.PathPatch(path, facecolor=color)
    return patch


def generate_image(
    ax: Axes,
    cls: int,
    location: np.ndarray,
    scale: int,
    rotation: float,
    color: np.ndarray,
    imsize: int = 32,
) -> None:
    color = color.astype(np.float32) / 255
    patch: patches.Patch
    if cls == 0:
        patch = get_diamond_patch(location, scale, rotation, color)
    elif cls == 1:
        patch = get_egg_patch(location, scale, rotation, color)
    elif cls == 2:
        patch = get_triangle_patch(location, scale, rotation, color)
    else:
        raise ValueError("Class does not exist.")

    ax.add_patch(patch)
    ax.set_xticks([])  # type: ignore
    ax.set_yticks([])  # type: ignore
    ax.grid(False)
    ax.set_xlim(0, imsize)
    ax.set_ylim(0, imsize)


def generate_scale(n_samples: int, min_val: int, max_val: int) -> np.ndarray:
    assert max_val > min_val
    return np.random.randint(min_val, max_val + 1, n_samples)


def generate_color(
    n_samples: int, min_lightness: int = 0, max_lightness: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    assert 0 <= max_lightness <= 256
    hls = np.random.randint(
        [0, min_lightness, 0],
        [181, max_lightness, 256],
        size=(1, n_samples, 3),
        dtype=np.uint8,
    )
    rgb = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)[0]  # type: ignore
    return rgb.astype(int), hls[0].astype(int)


def generate_rotation(n_samples: int) -> np.ndarray:
    rotations = np.random.rand(n_samples) * 2 * np.pi
    return rotations


def generate_location(
    n_samples: int, max_scale: int, imsize: int
) -> np.ndarray:
    assert max_scale <= imsize
    margin = max_scale // 2
    locations = np.random.randint(margin, imsize - margin, (n_samples, 2))
    return locations


def generate_class(n_samples: int) -> np.ndarray:
    return np.random.randint(3, size=n_samples)


if __name__=='__main__':
    dpi = 1
    imsize = 64
    fig, ax = plt.subplots(figsize=(imsize / dpi, imsize / dpi), dpi=dpi)
    ax = cast(plt.Axes, ax)
    generate_image(ax, 2, np.array([32., 32.]), 15, 3.14, np.array([255.,0.,0.]), imsize)
    ax.set_facecolor("black")
    plt.tight_layout(pad=0)
    plt.show()
