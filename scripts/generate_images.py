import matplotlib.pyplot as plt
from typing import cast

from Simple_Shapes_RL.utils import generate_image, generate_new_attributes


if __name__ == '__main__':

    attributes = generate_new_attributes()

    fig, ax = plt.subplots(figsize=(32, 32), dpi=1)
    ax = cast(plt.Axes, ax)

    generate_image(ax, attributes[0], attributes[1:3], attributes[3], attributes[4],
                    attributes[5:8], 32)

    ax.set_facecolor("black")
    plt.tight_layout(pad=0)

    plt.savefig('0.png', dpi=10)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(32, 32), dpi=1)
    ax = cast(plt.Axes, ax)

    generate_image(ax, attributes[0], [16, 16], attributes[3], 0,
                   attributes[5:8], 32)

    ax.set_facecolor("black")
    plt.tight_layout(pad=0)

    plt.savefig('end.png', dpi=10)
    plt.close(fig)