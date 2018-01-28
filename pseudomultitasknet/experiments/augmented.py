from pseudomultitasknet.utils import imshow

import torch

import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import augmnist


def main():
    dataloader = torch.utils.data.DataLoader(
        augmnist.AugMNIST(8, generate=True),
        batch_size=8,
        shuffle=False, num_workers=1
    )

    dataiter = iter(dataloader)
    images = dataiter.next()

    # images = images.view(-1, 1, 28, 28)
    print(images.size())

    imshow(torchvision.utils.make_grid(images, 8))
    plt.savefig('interpolation.pdf')

if __name__ == '__main__':
    main()
