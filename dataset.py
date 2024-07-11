from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms

class EMNIST(Dataset):
    def __init__(self, train=True):
        self.train = train
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda img: img.permute(0, 2, 1)),
            transforms.Normalize((0.5,), (0.5,)),
            self.Patchify(),
        ])
        target_transform = transforms.Lambda(
            lambda x: 0 if x == 11 else 1 if x == 14 else 2 if x == 15 else -1
        )
        dataset = datasets.EMNIST(
            root='./data',
            split='balanced',
            train=self.train,
            download=True,
            transform=transform,
            target_transform=target_transform
        )
        indicies = [i for i, (_, y) in enumerate(dataset) if y != -1]
        self.dataset = Subset(dataset, indicies)

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)

    class Patchify(object):
        def __init__(self, patch_size=7):
            self.patch_size = patch_size

        def __call__(self, img):
            c, h, w = img.shape
            img = img.unfold(1, self.patch_size, self.patch_size).unfold(2, self.patch_size, self.patch_size)
            img = img.permute(1, 2, 0, 3, 4)
            img = img.contiguous().view(-1, c, self.patch_size, self.patch_size)
            return img


if __name__ == "__main__":
    import torch
    from torchvision.utils import make_grid
    import matplotlib.pyplot as plt

    dataset = EMNIST(train=False)

    img_list = [[], [], []]
    for i, (X, y) in enumerate(dataset):
        X = make_grid(X, nrow=4, padding=0)
        img_list[y].append(X)

    for i in range(3):
        img = torch.stack(img_list[i])
        img = make_grid(img, nrow=16, padding=0)
        img = img.permute(1, 2, 0)
        img = img * 0.5 + 0.5
        fig, ax = plt.subplots(figsize=(64, 100))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.axis('off')
        ax.imshow(img)
        fig.savefig(f'data/{i}')
        plt.close(fig)