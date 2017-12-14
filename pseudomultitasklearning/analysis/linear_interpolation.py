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



# Prepare model

# model = train.main()
model = ReversibleMultiTaskNet()
model.load_state_dict(torch.load('experiments/mnist_ReversibleMultiTaskNet_2017-12-14_12-32-28/checkpoints/mnist_ReversibleMultiTaskNet_2017-12-14_12-58-02.dat'))
model.cuda()



# Interpolate features
steps = np.linspace(0, 1, 16)

for i in range(4):
    idx = torch.LongTensor(2).random_(0, ones.size(0))

    samples = ones.index_select(0, idx)

    # Prepare features
    features = model.no_class_forward(Variable(samples).cuda())

    interpolation = torch.stack([t*features.data[0]+(1-t)*features.data[1] for t in steps])

    generated = model.generate(Variable(interpolation).cuda())

    imshow(torchvision.utils.make_grid(generated.data.cpu(), 16))
    plt.savefig('interpolation{}.pdf'.format(i))
