import os

import numpy as np
import torch
import torchvision
from torchvision import transforms
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from pytorch_lightning import seed_everything
from typing import cast
from PIL import Image 
import matplotlib.backends.backend_agg as agg

from bim_gw.modules.domain_modules.vae import VAE as VAE_SS
from bim_gw.utils import get_args
from coco.domain_modules.mobile.vae import VAE as VAE_Fac
from Simple_Shapes_RL.utils import get_obs_space, get_action_space, generate_new_attributes, generate_new_target, generate_image

def explore_vae_SS(checkpoint):
    
    seed_everything(34) #5, 10, 26, 34

    device = torch.device("cuda")

    vae = (
        VAE_SS.load_from_checkpoint(
            checkpoint, strict=False
        )
        .to(device)
        .eval()
    )
    vae.freeze()

    fig, ax = plt.subplots(figsize=(32, 32), dpi=1)
    ax = cast(plt.Axes, ax)

    attributes = generate_new_attributes()

    attributes = np.array([1., 17., 16., 13., 2.626, 180., 79., 61., 35., 196., 160.])

    generate_image(ax, attributes[0], attributes[1:3], attributes[3], attributes[4],
                    attributes[5:8], 32)

    ax.set_facecolor("black")
    plt.tight_layout(pad=0)

    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    plt.close(fig)
    np_img = np.fromstring(buf, dtype=np.uint8).reshape(nrows, ncols, 3)
    img_tensor = torch.tensor(np_img/255.).permute(2, 0, 1).unsqueeze(0).float().to("cuda:0")

    z = vae.encode({"img": img_tensor})['z_img'].detach().cpu()[0]

    z_i_bis = torch.linspace(-2,2,9)

    z = z.unsqueeze(0).repeat(9,1)

    img_grid = torch.zeros(z.shape[1] * 9, img_tensor.shape[1], img_tensor.shape[2], img_tensor.shape[3])

    for i in range(z.shape[1]):
        z[:,i] += z_i_bis
        img = vae.decode({'z_img': z.to("cuda:0")})['img']
        img_grid[9*i:9*i+9] = img
    
    grid = torchvision.utils.make_grid(img_grid, nrow=9)
    torchvision.utils.save_image(grid, 'SS_VAE_explore.pdf')
    torchvision.utils.save_image(img_tensor, 'SS_original.pdf')


def interpolation_vae_SS(checkpoint):
    
    seed_everything(2)

    device = torch.device("cuda")

    vae = (
        VAE_SS.load_from_checkpoint(
            checkpoint, strict=False
        )
        .to(device)
        .eval()
    )
    vae.freeze()

    attributes = generate_new_attributes()
    attributes[0] = 0
    attributes[1] = 16
    attributes[2] = 16
    attributes[5:8] = np.array([255.,0.,0.])

    fig, ax = plt.subplots(figsize=(32, 32), dpi=1)
    ax = cast(plt.Axes, ax)
    generate_image(ax, attributes[0], attributes[1:3], attributes[3], attributes[4],
                    attributes[5:8], 32)
    ax.set_facecolor("black")
    plt.tight_layout(pad=0)
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    plt.close(fig)
    np_img = np.fromstring(buf, dtype=np.uint8).reshape(nrows, ncols, 3)
    img_tensor_1 = torch.tensor(np_img/255.).permute(2, 0, 1).unsqueeze(0).float().to("cuda:0")


    fig2, ax2 = plt.subplots(figsize=(32, 32), dpi=1)
    ax2 = cast(plt.Axes, ax2)
    generate_image(ax2, 2, [16,16], attributes[3], attributes[4],
                    np.array([0.,0.,255.]), 32)
    ax2.set_facecolor("black")
    plt.tight_layout(pad=0)
    fig2.canvas.draw()
    buf = fig2.canvas.tostring_rgb()
    ncols, nrows = fig2.canvas.get_width_height()
    plt.close(fig2)
    np_img = np.fromstring(buf, dtype=np.uint8).reshape(nrows, ncols, 3)
    img_tensor_2 = torch.tensor(np_img/255.).permute(2, 0, 1).unsqueeze(0).float().to("cuda:0")

    z_1 = vae.encode({"img": img_tensor_1})['z_img'].detach().cpu()[0]
    z_2 = vae.encode({"img": img_tensor_2})['z_img'].detach().cpu()[0]

    z_1_2 = torch.zeros(z_1.shape[0] , 20)
    for i in range(z_1.shape[0]):
        z_1_2[i,:] = torch.linspace(z_1[i], z_2[i], 20)

    z_1_2 = z_1_2.T

    img_grid = torch.zeros(20, img_tensor_1.shape[1], img_tensor_1.shape[2], img_tensor_1.shape[3])

    for i in range(20):
        img = vae.decode({'z_img': z_1_2[i].unsqueeze(0).to("cuda:0")})['img']
        img_grid[i] = img
    
    grid = torchvision.utils.make_grid(img_grid, nrow=20)
    torchvision.utils.save_image(grid, 'SS_VAE_interpolate.pdf')
    torchvision.utils.save_image(img_tensor_1, 'SS_begin.pdf')
    torchvision.utils.save_image(img_tensor_2, 'SS_end.pdf')


def explore_vae_Fac(checkpoint):
    
    seed_everything(1)

    device = torch.device("cuda")

    vae = (
        VAE_Fac.load_from_checkpoint(
            checkpoint, strict=False
        )
        .to(device)
        .eval()
    )
    vae.freeze()

    transform = transforms.Compose([transforms.ToTensor()])
    img = Image.open('/home/leopold/HD2/factory/mobile/table/test/0.png')
    img_tensor = transform(img).unsqueeze(0).to(device)

    z = vae.encode({"img": img_tensor})['z_img'].detach().cpu()[0]

    z_i_bis = torch.linspace(-2,2,9)

    z = z.unsqueeze(0).repeat(9,1)

    img_grid = torch.zeros(z.shape[1] * 9, img_tensor.shape[1], img_tensor.shape[2], img_tensor.shape[3])

    for i in range(z.shape[1]):
        z[:,i] += z_i_bis
        img = vae.decode(z.to("cuda:0"))['img']
        img_grid[9*i:9*i+9] = img
    
    grid = torchvision.utils.make_grid(img_grid, nrow=9)
    torchvision.utils.save_image(grid, 'Fac_VAE_explore.pdf')
    torchvision.utils.save_image(img_tensor, 'Fac_original.pdf')


if __name__ == '__main__':

    ckpt_SS = 'RL_Simple_Shapes/Simple_Shapes_RL/822888/epoch=282-step=1105680.ckpt'
    ckpt_Fac = '/mnt/HD2/leopold/checkpoints/VAE_tables_mobile/train_vae/VAE-240/checkpoints/epoch=721-step=1128486.ckpt'
    explore_vae_SS(ckpt_SS)
    explore_vae_Fac(ckpt_Fac)
    interpolation_vae_SS(ckpt_SS)