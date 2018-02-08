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
