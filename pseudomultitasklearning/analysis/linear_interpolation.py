from pseudomultitasklearning import train
from pseudomultitasklearning.utils import imshow
from pseudomultitasklearning.modules.reversible import ReversibleMultiTaskNet

import torch

import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# Prepare data

dataloader = train.load_mnist(10000, shuffle=False, train=False, num_workers=0)

dataiter = iter(dataloader)
images, labels = dataiter.next()

ones = images.index_select(0, torch.nonzero(labels == 1).squeeze())

idx = torch.LongTensor(2).random_(0, ones.size(0))

samples = ones.index_select(0, idx)


# Prepare model

model = train.main()


# Prepare features
features = model.no_class_forward(Variable(samples).cuda())


# Interpolate features
steps = np.linspace(0, 1, 16)

interpolation = torch.stack([t*features.data[0]+(1-t)*features.data[1] for t in steps])

generated = model.generate(Variable(interpolation).cuda())

imshow(torchvision.utils.make_grid(generated.cpu(), 16))
plt.savefig('interpolation.pdf')
