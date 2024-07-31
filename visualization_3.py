import numpy as np
import torch
import torchvision
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


img = np.array([range(100)])
fig, ax = plt.subplots(figsize=(20, 1))
fig, ax = plt.subplots()
ax.axis('off')
ax.imshow(img, cmap='Spectral_r')
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
fig.savefig('tmp')
plt.close(fig)