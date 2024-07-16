import argparse

import numpy as np
import torch
import torch.optim as optim
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from dataset import EMNIST
from model import Attention, Additive

# Training settings
parser = argparse.ArgumentParser(description='EMNIST')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                    help='learning rate (default: 0.0005)')
parser.add_argument('--reg', type=float, default=10e-5, metavar='R',
                    help='weight decay')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--batch_size', type=int, default=4, metavar='B',
                    help='batch_size (default: 4)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--model', type=str, default='attention', choices=['attention', 'additive'],
                    help='model (default: attention)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    print('\nGPU is ON!')

print('Load Train and Test Set')
loader_kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

train_loader = torch.utils.data.DataLoader(EMNIST(train=True),
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           **loader_kwargs)

test_loader = torch.utils.data.DataLoader(EMNIST(train=False),
                                          batch_size=1,
                                          shuffle=False,
                                          **loader_kwargs)

print('Init Model')
if args.model=='attention':
    model = Attention()
elif args.model=='additive':
    model = Additive()
if args.cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)


def train(epoch):
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss()
    train_loss = 0.
    train_error = 0.

    for i, (X_batch, y_batch) in enumerate(train_loader):
        if args.cuda:
            X_batch, y_batch = X_batch.cuda(), y_batch.cuda()

        optimizer.zero_grad()

        y_proba_list = []
        y_hat_list = []
        for j, (X, y) in enumerate(zip(X_batch, y_batch)):
            X = X.unsqueeze(0)
            y = y.unsqueeze(0)
            y_proba, y_hat, _ = model(X)
            y_proba_list.append(y_proba)
            y_hat_list.append(y_hat)
        y_proba = torch.cat(y_proba_list, dim=0)
        y_hat = torch.cat(y_hat_list, dim=0)

        loss = loss_fn(y_proba, y_batch)
        loss.backward()

        optimizer.step()

        train_loss += loss.detach().cpu().item()
        train_error += 1. - (y_hat == y_batch).detach().cpu().count_nonzero().item() / args.batch_size

    train_loss /= len(train_loader)
    train_error /= len(train_loader)

    print('Epoch: {:2d}, Loss: {:.4f}, Train error: {:.4f}'.format(epoch, train_loss, train_error))


def test():
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    test_loss = 0.
    test_error = 0.

    with torch.no_grad():
        cnt = np.zeros(3, dtype=int)
        for i, (X, y) in enumerate(test_loader):
            if args.cuda:
                X, y = X.cuda(), y.cuda()

            y_proba, y_hat, A = model(X)
            loss = loss_fn(y_proba, y)
            test_loss += loss.detach().cpu().item()
            test_error += 1. - (y_hat == y).detach().cpu().count_nonzero().item()

            y = y.detach().cpu()[0]
            k = cnt[y]
            if k < 10:
                X = X.detach().cpu()[0]
                A = A.detach().cpu()[0]
                y_hat = y_hat.detach().cpu().int()[0]

                if args.model == 'attention':
                    save_result(X, A, title=f'$y = {y}, \\hat{{y}} = {y_hat}$', filename=f'img_{y}_{k}')
                elif args.model == 'additive':
                    A = torch.permute(A, (1, 0))
                    for l in range(3):
                        save_result(X, A[l], title=f'$y = {y}, \\hat{{y}} = {y_hat}, y\' = {l}$', filename=f'img_{y}_{k}_{l}')
                cnt[y] += 1

    test_error /= len(test_loader)
    test_loss /= len(test_loader)

    print('Test Set, Loss: {:.4f}, Test error: {:.4f}'.format(test_loss, test_error))


def save_result(X, A, title=None, path=f'./img/{args.model}/', filename='img', mean=torch.tensor([0.5]), std=torch.tensor([0.5])):
    X = make_grid(X, nrow=4, padding=0)
    X = X * std + mean
    X = torch.permute(X, (1, 2, 0))
    A = A.contiguous().view(4, 4)

    fig, ax = plt.subplots(figsize=(3.6, 4))
    if title is not None:
        fig.suptitle(title)
    ax.axis('off')
    ax.imshow(X)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    if args.model == 'attention':
        ax.imshow(A, cmap='bwr', alpha=0.5, extent=[*xlim, *ylim])
    elif args.model == 'additive':
        ax.imshow(A, cmap='bwr', alpha=0.5, vmin=0, vmax=1, extent=[*xlim, *ylim])
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.9)
    fig.savefig(path + filename)
    plt.close(fig)


if __name__ == "__main__":
    print('Start Training')
    for epoch in range(1, args.epochs + 1):
        train(epoch)
    print('Start Testing')
    test()
