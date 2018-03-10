from pseudomultitasknet.data import load_mnist
from pseudomultitasknet.utils import imshow
from pseudomultitasknet import PseudoMultiTaskNet

import torch

import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# Prepare data

dataloader = load_mnist(10000, shuffle=False, train=False, num_workers=0)

dataiter = iter(dataloader)
images, labels = dataiter.next()

ones = images.index_select(0, torch.nonzero(labels == 1).squeeze())



# Prepare model

# model = train.main()
model = PseudoMultiTaskNet()
model.load_state_dict(torch.load('mnist_ReversibleMultiTaskNet.dat'))
model.cuda(0)



# Interpolate features
steps = np.linspace(0, 1, 16)

for i in range(4):
    idx = torch.LongTensor(2).random_(0, ones.size(0))

    samples = ones.index_select(0, idx)

    # Prepare features
    features = model.inversible_forward(Variable(samples).cuda())

    interpolation = torch.stack([t*features.data[0]+(1-t)*features.data[1] for t in steps], dim=0)

    generated = model.generate(interpolation.cuda())

    regenerated = model.generate(features)

    # features_ = model.orthogonal_forward(Variable(samples).cuda())

    # interpolation_ = torch.stack([t*features_.data[0]+(1-t)*features_.data[1] for t in steps], dim=0)

    # generated_ = model.generate(interpolation_.cuda())

    imshow(torchvision.utils.make_grid(generated.data.cpu(), 16))
    plt.savefig('interpolation{}.pdf'.format(i))

    # imshow(torchvision.utils.make_grid(generated_.data.cpu(), 16))
    # plt.savefig('interpolation{}_.pdf'.format(i))

    imshow(torchvision.utils.make_grid(samples, 2))
    plt.savefig('originals{}.pdf'.format(i))

    imshow(torchvision.utils.make_grid(regenerated.data.cpu(), 2))
    plt.savefig('regenerated{}.pdf'.format(i))
