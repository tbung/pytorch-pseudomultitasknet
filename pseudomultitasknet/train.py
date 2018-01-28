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

# from pytorch_fft.fft.autograd import Rfft2d

from pseudomultitasknet import PseudoMultiTaskNet
import augmnist

parser = argparse.ArgumentParser()
# parser.add_argument("--model", metavar="NAME",
#                     help="what model to use")
parser.add_argument("--load", metavar="PATH",
                    help="load a previous model state")
parser.add_argument("-e", "--evaluate", action="store_true",
                    help="evaluate model on validation set")
parser.add_argument("--batch-size", default=256, type=int,
                    help="size of the mini-batches")
parser.add_argument("--epochs", default=10, type=int,
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

# fft = Rfft2d()


def main():
    global best_acc

    args = parser.parse_args()

    model = PseudoMultiTaskNet()

    exp_id = "mnist_{0}_{1:%Y-%m-%d}_{1:%H-%M-%S}".format(model.name,
                                                          datetime.now())

    path = os.path.join("./saves/", exp_id, "cmd.sh")
    if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

    with open(path, 'w') as f:
        f.write(' '.join(sys.argv))

    # if CUDA:
    model.cuda(0)

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

    optimizer2 = optim.SGD(model.parameters(), lr=1e-6,
                          momentum=0.9, weight_decay=args.weight_decay)

    scheduler = StepLR(optimizer, step_size=args.epochs//3, gamma=0.1)

    print("Prepairing data...")

    # Load data
    trainloader = load_mnist(args.batch_size, train=True)
    valloader = load_mnist(args.batch_size, train=False)
    aug_bs = 16
    augloader =  load_augmented(aug_bs)

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

        train_augmented(epoch, model, criterion, optimizer2,
                        augloader, aug_bs, args.clip, trainer)

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


def load_mnist(batch_size, shuffle=True, train=False, num_workers=2):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = torchvision.datasets.MNIST(
        root='./data', train=train,
        download=True, transform=transform
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle, num_workers=num_workers
    )

    return dataloader

def load_augmented(batch_size, num_workers=2):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = augmnist.AugMNIST(
        batch_size=batch_size,
        download=False,
        generate=True,
        transform=transform
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False, num_workers=num_workers
    )

    return dataloader


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
        sparsity = torch.norm(outputs[3], 1)

        loss =  criterion(likelihoods, labels) + 0.01*disagreement + 0.0001*sparsity
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

def train_backwards(epoch, model, criterion, optimizer, trainloader, clip, trainer):
    model.train()
    # train_loss = 0
    # correct = 0
    # total = 0
    t = tqdm(trainloader, ascii=True, desc='{}'.format(epoch).rjust(3))
    for i, data in enumerate(t):
        inputs, labels = data

        if CUDA:
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

        # outputs = model.no_class_forward(inputs)
        outputs = model.orthogonal_forward(inputs)

        optimizer.zero_grad()

        # regenerated = torch.cat(fft(model.generate(outputs)), dim=1)
        regenerated = model.generate(outputs)

        loss = F.l1_loss(inputs, regenerated)

        if np.isnan(loss.data[0]):
            raise ValueError("NaN Loss")

        # graph = visualize.make_dot(loss)
        # graph.format = 'svg'
        # graph.render(view=False)
        # break

        loss.backward()

        # Free the memory used to store activations
        model.free()

        if clip > 0:
            torch.nn.utils.clip_grad_norm(model.parameters(), clip)

        optimizer.step()


def train_augmented(epoch, model, criterion, optimizer, trainloader, batch_size, clip, trainer):
    model.train()
    t = tqdm(trainloader, ascii=True, desc='{}'.format(epoch).rjust(3))
    for i, inputs in enumerate(t):
        if CUDA:
            inputs = Variable(inputs).cuda()

        outputs = model.inversible_forward(
            torch.stack((inputs[0], inputs[-1]), dim=0)
        )

        steps = np.linspace(0, 1, 16)
        interpolation = torch.stack([(1-t)*outputs.data[0]+t*outputs.data[1] for t in steps], dim=0)

        generated = model.generate(Variable(interpolation.cuda()))

        optimizer.zero_grad()

        loss = F.l1_loss(inputs, generated)

        if np.isnan(loss.data[0]):
            raise ValueError("NaN Loss")

        # graph = visualize.make_dot(loss)
        # graph.format = 'svg'
        # graph.render(view=False)
        # break

        loss.backward()

        # Free the memory used to store activations
        model.free()

        if clip > 0:
            torch.nn.utils.clip_grad_norm(model.parameters(), clip)

        optimizer.step()

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
