import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms


class MyDataset(Dataset):
    def __init__(self, train=True):
        self.train = train
        self.num = 6000 if self.train else 3

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
            torch.stack([self.img[0]] * 12 + [self.img[1]] * 2 + [self.img[2]] * 2, dim = 0)
        ]

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        return self.X[idx % 3], idx % 3


if __name__ == "__main__":
    import torch
    from torchvision.utils import make_grid
    import matplotlib.pyplot as plt

    dataset = MyDataset(train=False)
    for i in range(3):
        img = make_grid(dataset.X[i], nrow=4, padding=0)
        img = img.permute(1, 2, 0)
        img = img * 0.5 + 0.5
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.axis('off')
        ax.imshow(img)
        fig.savefig(f'data/{i}')
        plt.close(fig)