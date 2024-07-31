import numpy as np
import torch
import torchvision
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


for i in range(4):
    img_list = []
    for j in range(10):
        img_list.append(plt.imread(f'./img/attention/img_{i}_{j}.png'))
    img = np.stack(img_list)
    img = torch.from_numpy(img)
    img = torch.permute(img, (0, 3, 1, 2))
    img = torchvision.utils.make_grid(img, nrow=5, padding=0)
    img = torch.permute(img, (1, 2, 0))

    fig, ax = plt.subplots(figsize=(18, 8))
    ax.axis('off')
    ax.imshow(img)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(f'./visualization/attention/visualization_{i}')
    plt.close(fig)


for i in range(4):
    for k in range(4):
        img_list = []
        for j in range(10):
            img_list.append(plt.imread(f'./img/additive/img_{i}_{j}_{k}.png'))
        img = np.stack(img_list)
        img = torch.from_numpy(img)
        img = torch.permute(img, (0, 3, 1, 2))
        img = torchvision.utils.make_grid(img, nrow=5, padding=0)
        img = torch.permute(img, (1, 2, 0))

        fig, ax = plt.subplots(figsize=(18, 8))
        ax.axis('off')
        ax.imshow(img)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        fig.savefig(f'./visualization/additive/visualization_{i}_{k}')
        plt.close(fig)

exit()
for i in range(4):
    img_list = []
    for j in range(3):
        img_list.append(plt.imread(f'./img/attention/img_{i}_{j}.png'))
    img = np.stack(img_list)
    img = torch.from_numpy(img)
    img = torch.permute(img, (0, 3, 1, 2))
    # img = torchvision.utils.make_grid(img, nrow=3, padding=0)
    img = torchvision.utils.make_grid(img, nrow=3)
    img = torch.permute(img, (1, 2, 0))

    fig, ax = plt.subplots(figsize=(10.8, 4))
    ax.axis('off')
    ax.imshow(img)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(f'./visualization/attention/visualization_{i}')
    plt.close(fig)


for i in range(4):
    for k in range(4):
        img_list = []
        for j in range(3):
            img_list.append(plt.imread(f'./img/additive/img_{i}_{j}_{k}.png'))
        img = np.stack(img_list)
        img = torch.from_numpy(img)
        img = torch.permute(img, (0, 3, 1, 2))
        # img = torchvision.utils.make_grid(img, nrow=3, padding=0)
        img = torchvision.utils.make_grid(img, nrow=3)
        img = torch.permute(img, (1, 2, 0))

        fig, ax = plt.subplots(figsize=(10.8, 4))
        ax.axis('off')
        ax.imshow(img)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        fig.savefig(f'./visualization/additive/visualization_{i}_{k}')
        plt.close(fig)

exit()