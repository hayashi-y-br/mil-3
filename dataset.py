import random

import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms


class MyDataset(Dataset):
    def __init__(self, train=True):
        self.train = train
        self.num = 6000 if self.train else 1200

        img_list = [
            torch.Tensor([[[0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0]]]),
            torch.Tensor([[[0, 0, 1, 1, 1, 0, 0],
                           [0, 0, 1, 1, 1, 0, 0],
                           [1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1],
                           [0, 0, 1, 1, 1, 0, 0],
                           [0, 0, 1, 1, 1, 0, 0]]]),
            torch.Tensor([[[0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0],
                           [1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1],
                           [0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0]]])
        ]
        self.transform = transforms.Compose([
            transforms.Normalize((0.5,), (0.5,)),
        ])
        self.img = []
        for i in img_list:
            self.img.append(self.transform(i))

        self.X = [
            torch.stack([self.img[0]] * 16, dim = 0),
            torch.stack([self.img[0]] * 12 + [self.img[1]] * 4, dim = 0),
            torch.stack([self.img[0]] * 12 + [self.img[2]] * 4, dim = 0),
            torch.stack([self.img[0]] * 12 + [self.img[1]] * 2 + [self.img[2]] * 2, dim = 0)
        ]
        
        random.seed(1 if self.train else 0)
        self.data = [0] * self.num
        for i in range(self.num):
            n = 0
            m = 0
            if i % 4 == 1:
                n = random.randint(1, 10)
            elif i % 4 == 2:
                m = random.randint(1, 10)
            elif i % 4 == 3:
                n = random.randint(1, 5)
                m = random.randint(1, 5)
            self.data[i] = [100 - n - m, n, m]


    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        return torch.stack([self.img[1]] * self.data[idx][1] + [self.img[0]] * self.data[idx][0] + [self.img[2]] * self.data[idx][2], dim = 0), idx % 4


if __name__ == "__main__":
    import torch
    from torchvision.utils import make_grid
    import matplotlib.pyplot as plt

    dataset = MyDataset(train=False)

    for i in range(3):
        img = dataset.img[i]
        img = img.permute(1, 2, 0)
        img = img * 0.5 + 0.5
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.axis('off')
        ax.imshow(img, cmap='gray')
        fig.savefig(f'data/{i}')
        plt.close(fig)
    exit()

    for i in range(100):
        img = make_grid(dataset.__getitem__(i)[0], nrow=10, padding=0)
        img = img.permute(1, 2, 0)
        img = img * 0.5 + 0.5
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.axis('off')
        ax.imshow(img)
        fig.savefig(f'data/{i}')
        plt.close(fig)