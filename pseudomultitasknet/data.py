import torch
import torchvision
import torchvision.transforms as transforms

import augmnist


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


def load_cifar10(batch_size, shuffle=True, train=False, num_workers=2):
    if train:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

    dataset = torchvision.datasets.CIFAR10(
        root='./data', train=train,
        download=True, transform=transform
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle, num_workers=num_workers
    )

    return dataloader


def load_augmented(batch_size, interpolation_size, num_workers=2):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = augmnist.AugMNIST(
        batch_size=interpolation_size,
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
