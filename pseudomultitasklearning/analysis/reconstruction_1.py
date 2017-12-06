from pseudomultitasklearning import train_mnist
from pseudomultitasklearning.modules.reversible import ReversibleMultiTaskNet

import torch

import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable

import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA

# Load data
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

trainset = torchvision.datasets.MNIST(
    root='./data', train=True,
    download=True, transform=transform_train
)

trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=16,
                                          shuffle=True, num_workers=2)

trainloader_pca = torch.utils.data.DataLoader(trainset,
                                          batch_size=10000,
                                          shuffle=False, num_workers=2)

def imshow(img):
    plt.figure(tight_layout=True)
    img = (img - img.min())/(img.max()-img.min())
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


def unsqueeze(x):
    return x.view(-1, x.size(1)//4, x.size(2)*2, x.size(3)*2)


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# save originals
imshow(torchvision.utils.make_grid(images))
plt.savefig('originals.svg',bbox_inches='tight')

# generate and look at feature space
m = train_mnist.main()
out = m.no_class_forward(Variable(images).cuda())
out2 = unsqueeze(unsqueeze(out))
imshow(torchvision.utils.make_grid(out2.cpu().data))
plt.savefig('featurespace.svg',bbox_inches='tight')

# invert net, regenerate input
out3 = m.generate(out)
imshow(torchvision.utils.make_grid(out3.cpu()))
plt.savefig('regenerated.svg',bbox_inches='tight')

dataiter_pca = iter(trainloader_pca)
images, labels = dataiter_pca.next()
imshow(torchvision.utils.make_grid(images[:5]))
plt.savefig('originals_pca.svg',bbox_inches='tight')
data = m.no_class_forward(Variable(images).cuda()).cpu().data

data = data.view(data.size(0), -1)
for i in range(10):
    print(data[:,i].min(), data[:,i].max())
    _, indices = torch.sort(data[:,i].squeeze())

    imshow(torchvision.utils.make_grid(images[indices][::50]))
    plt.savefig('sorted{}.svg'.format(i),bbox_inches='tight')

for i in range(10,15):
    print(data[:,i].min(), data[:,i].max())
    _, indices = torch.sort(data[:,i].squeeze())

    imshow(torchvision.utils.make_grid(images[indices][::50]))
    plt.savefig('sorted{}.svg'.format(i),bbox_inches='tight')

seed = torch.rand(1,16*7*7)
seed[0,0] = 100
seed[0,1:10] = -10
seed = seed.view(1,16,7,7)
imshow(torchvision.utils.make_grid(m.generate(Variable(seed).cuda()).cpu()))
plt.savefig('generated.svg',bbox_inches='tight')

pca = PCA(15)
pca.fit(data)
trans = pca.transform(data)
# print(labels[0], trans)
# print(labels[1], pca.transform(data[1].reshape(1,-1)))

regenerated = pca.inverse_transform(trans[:5])
regenerated.shape = -1, 16, 7, 7
regenerated = m.generate(torch.Tensor(regenerated).cuda())
imshow(torchvision.utils.make_grid(regenerated.cpu()))
plt.savefig('pca.svg',bbox_inches='tight')

for i in range(15):
    print(trans[:,i].min(), trans[:,i].max())
    indices = np.argsort(trans[:,i])

    imshow(torchvision.utils.make_grid(images[torch.Tensor(indices).long()][::50]))
    plt.savefig('sorted_pca{}.svg'.format(i),bbox_inches='tight')
