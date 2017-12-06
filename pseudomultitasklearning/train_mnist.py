from datetime import datetime

import os
import sys
import argparse

from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

from .modules.naive_bayes import GaussianNaiveBayes
from .modules.reversible import ReversibleMultiTaskNet

class NaiveBayesNet(nn.Module):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.name = "naive-bayes-net"
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.nb  = GaussianNaiveBayes(10,10)
        self.sm  = nn.LogSoftmax(dim=0)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        prob_nb = torch.exp(self.nb(x))
        prob_sm = torch.exp(self.sm(x))
        return torch.log(0.5*(prob_nb + prob_sm))

parser = argparse.ArgumentParser()
# parser.add_argument("--model", metavar="NAME",
#                     help="what model to use")
parser.add_argument("--load", metavar="PATH",
                    help="load a previous model state")
parser.add_argument("-e", "--evaluate", action="store_true",
                    help="evaluate model on validation set")
parser.add_argument("--batch-size", default=256, type=int,
                    help="size of the mini-batches")
parser.add_argument("--epochs", default=50, type=int,
                    help="number of epochs")
parser.add_argument("--lr", default=0.1, type=float,
                    help="initial learning rate")
parser.add_argument("--clip", default=0, type=float,
                    help="maximal gradient norm")
parser.add_argument("--weight-decay", default=1e-4, type=float,
                    help="weight decay factor")
parser.add_argument("--stats", action="store_true",
                    help="record and plot some stats")
parser.add_argument("--trainer", action="store_true", default=False,
                    help="should we use the outputs of a pretrained network for training")
parser.add_argument("--loss", default="NLLLoss",
                    help="which loss function to use")


# Check if CUDA is avaliable
CUDA = torch.cuda.is_available()

best_acc = 0


def main():
    global best_acc

    args = parser.parse_args()

    model = ReversibleMultiTaskNet()

    exp_id = "mnist_{0}_{1:%Y-%m-%d}_{1:%H-%M-%S}".format(model.name,
                                                          datetime.now())

    path = os.path.join("./experiments/", exp_id, "cmd.sh")
    if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

    with open(path, 'w') as f:
        f.write(' '.join(sys.argv))

    if CUDA:
        model.cuda()

    if args.load is not None:
        load(model, args.load)

    if args.trainer:
        # Load trainer, wherever it may come from
        trainer = None
    else:
        trainer = None

    criterion = getattr(nn, args.loss)()

    optimizer = optim.SGD(model.parameters(), lr=args.lr*10,
                          momentum=0.9, weight_decay=args.weight_decay)

    scheduler = StepLR(optimizer, step_size=args.epochs//3, gamma=0.1)

    print("Prepairing data...")

    # Load data
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    trainset = torchvision.datasets.MNIST(
        root='./data', train=True,
        download=True, transform=transform_train
    )

    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=args.batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(
        root='./data', train=False,
        download=True, transform=transform_test
    )

    valloader = torch.utils.data.DataLoader(testset,
                                            batch_size=args.batch_size,
                                            shuffle=False, num_workers=2)

    if args.evaluate:
        print("\nEvaluating model...")
        acc = validate(model, valloader)
        print('Accuracy: {}%'.format(acc))
        return

    if args.stats:
        losses = []
        taccs = []
        vaccs = []

    print("\nTraining model...")
    for epoch in range(args.epochs):
        scheduler.step()
        loss, train_acc = train(epoch, model, criterion, optimizer,
                                trainloader, args.clip, trainer)
        val_acc = validate(model, valloader)

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model, exp_id)
        print('Accuracy: {}%'.format(val_acc))

        if args.stats:
            losses.append(loss)
            taccs.append(train_acc)
            vaccs.append(val_acc)

    save_checkpoint(model, exp_id)

    if args.stats:
        path = os.path.join("./experiments/", exp_id, "stats/{}.dat")
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        with open(path.format('loss'), 'w') as f:
            for i in losses:
                f.write('{}\n'.format(i))

        with open(path.format('taccs'), 'w') as f:
            for i in taccs:
                f.write('{}\n'.format(i))

        with open(path.format('vaccs'), 'w') as f:
            for i in vaccs:
                f.write('{}\n'.format(i))

    return model


def train(epoch, model, criterion, optimizer, trainloader, clip, trainer):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    t = tqdm(trainloader, ascii=True, desc='{}'.format(epoch).rjust(3))
    for i, data in enumerate(t):
        inputs, labels = data

        if CUDA:
            inputs, labels = inputs.cuda(), labels.cuda()

        inputs, labels = Variable(inputs), Variable(labels)

        if trainer is not None:
            labels = trainer(inputs)

        optimizer.zero_grad()

        outputs = model(inputs)
        likelihoods = torch.log((outputs[0]+outputs[1]+outputs[2])/3)
        disagreement = torch.sum(torch.abs(outputs[0] - outputs[1]))

        loss =  criterion(likelihoods, labels) + 1*disagreement
        if np.isnan(loss.data[0]):
            raise ValueError("NaN Loss")
        loss.backward()

        # Free the memory used to store activations
        model.free()

        if clip > 0:
            torch.nn.utils.clip_grad_norm(model.parameters(), clip)
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(likelihoods.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels.data).cpu().sum()
        acc = 100 * correct / total

        t.set_postfix(loss='{:.3f}'.format(train_loss/(i+1)).ljust(3),
                      acc='{:2.1f}%'.format(acc).ljust(6))

    return train_loss, acc


def validate(model, valloader):
    correct = 0
    total = 0

    model.eval()

    for data in valloader:
        images, labels = data
        if CUDA:
            images, labels = images.cuda(), labels.cuda()
        outputs = model(Variable(images))
        likelihoods = torch.log((outputs[0]+outputs[1]+outputs[2])/3)

        # Free the memory used to store activations
        model.free()

        _, predicted = torch.max(likelihoods.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    acc = 100 * correct / total

    return acc


def load(model, path):
    model.load_state_dict(torch.load(path))


def save_checkpoint(model, exp_id):
    path = os.path.join(
        "experiments", exp_id, "checkpoints",
        "mnist_{0}_{1:%Y-%m-%d}_{1:%H-%M-%S}.dat".format(model.name,
                                                         datetime.now()))
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    torch.save(model.state_dict(), path)


if __name__ == "__main__":
    main()
