import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.M = 500
        self.L = 128

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(20, 50, kernel_size=3),
            nn.ReLU()
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 3 * 3, self.M),
            nn.ReLU()
        )

        self.attention = nn.Sequential(
            nn.Linear(self.M, self.L),  # matrix V
            nn.Tanh(),
            nn.Linear(self.L, 1)  # vector w
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.M, 4),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        H = H.contiguous().view(-1, 50 * 3 * 3)
        H = self.feature_extractor_part2(H)  # KxM

        A = self.attention(H)  # Kx1
        A = torch.transpose(A, 1, 0)  # 1xK
        A = F.softmax(A, dim=1)  # softmax over K

        Z = torch.mm(A, H)  # 1xM

        y_proba = self.classifier(Z)
        y_hat = torch.argmax(y_proba, dim=1).float()

        return y_proba, y_hat, A


class Additive(nn.Module):
    def __init__(self):
        super(Additive, self).__init__()
        self.M = 500
        self.L = 128

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(20, 50, kernel_size=3),
            nn.ReLU()
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 3 * 3, self.M),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.M, self.L),  # matrix V
            nn.Tanh(),
            nn.Linear(self.L, 1)  # vector w
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.M, 4),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        H = H.contiguous().view(-1, 50 * 3 * 3)
        H = self.feature_extractor_part2(H)  # KxM

        A = self.attention(H)  # Kx1
        A = F.softmax(A, dim=0)  # softmax over K

        Z = torch.mul(A, H)  # KxM

        P = self.classifier(Z)  # Kx3
        P = P.unsqueeze(0)  # 1xKx3

        y_proba = torch.mean(P, dim=1)  # 1x3
        y_hat = torch.argmax(y_proba, dim=1).float()

        return y_proba, y_hat, torch.transpose(A, 1, 0), P