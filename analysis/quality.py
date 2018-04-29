from pseudomultitasknet.data import load_augmented, load_mnist
from pseudomultitasknet import PseudoMultiTaskNet, PseudoMultiTaskNetMult

import torch

import torchvision

from torch.autograd import Variable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path
from collections import OrderedDict

from tqdm import tqdm

import fire
import json


epoch_num = 17


def shear(image, value, axis):
    image = image.squeeze()
    base = image.min()
    coordinates = torch.nonzero(image!=base).t().float()

    fill = image[image!=base].clone()

    rangey,rangex = image.size()

    half = image.size(axis) // 2
    coordinates[1 - axis] = torch.ceil(coordinates[1 - axis] + (coordinates[axis]-half)*value)

    if coordinates.max() < rangex and coordinates.min() >= 0:
        result = torch.sparse.FloatTensor(coordinates.long(), fill, image.size()).to_dense()
        result[result == 0] = base
        return result
    else:
        # print('Overflow')
        return image

def load_checkpoints(epoch_short, epoch_long):
    basepath = Path.home() / "Projects" / "pytorch-pseudomultitasknet" / "runs"
    checkpoint_paths = basepath.glob("**/checkpoints")
    models = OrderedDict({
        "sm_nb": None,
        "class": None,
        "class_sp": None,
        "class_inter": None,
        "class_inter_ba": None,
        "class_sp_inter": None,
        "class_sp_inter_ba": None,
        "class_sp_inter_ba_small": None,
        "class_sp_inter_ba_long": None,
        "class_sp_inter_ba_mult": None,
        "class_sp_inter_ba_mult_long": None
    })

    for path in checkpoint_paths:
        with open(path / ".." / "records.json") as f:
            obj = json.load(f)
        name = obj["name"]

        if "long" in name:
            epoch = epoch_long
        else:
            epoch = epoch_short

        if "mult" in name:
            model = PseudoMultiTaskNetMult()
        elif "small" in name:
            model = PseudoMultiTaskNet(small=True)
        else:
            model = PseudoMultiTaskNet()
        model.load_state_dict(torch.load(path / f"checkpoint_{epoch}.dat"))
        model.cuda(0)
        models[name] = model

    models = {k.replace("sp", "tr"): v for k, v in models.items()}

    return models


def generate_plots(epoch_short, epoch_long):
    models = load_checkpoints(epoch_short, epoch_long)

    # dataloader = load_augmented(8*16, 16, num_workers=2)
    dataloader = load_mnist(1, shuffle=True, train=False, num_workers=2)
    dataiter = iter(dataloader)
    image, label = dataiter.next()
    T = np.linspace(-0.5, 0.5, 16)

    def generate_interpolation():
        for t in T:
            yield shear(image, t, 0)

    images = torch.stack(list(generate_interpolation()),
                           dim=0)
    images = images.view(16, 1, 28, 28)

    heights = [images.size(2)] * (len(models) + 1)

    fig_width = 8.0
    fig_height = fig_width * sum(heights) / (images.size(3) * 16)

    steps = np.linspace(0, 1, 16)

    fig, axes = plt.subplots(
        len(models) + 1, 1,
        figsize=(fig_width + 4, fig_height),
        gridspec_kw={'height_ratios': heights}
    )
    fig.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0,
                        top=1)

    img = torchvision.utils.make_grid(images.data.cpu(), 16)
    img = (img - img.min())/(img.max()-img.min())
    npimg = img.numpy()

    axes[0].imshow(np.transpose(npimg, (1, 2, 0)))
    axes[0].text(-7, 17, "original", horizontalalignment='right')
    axes[0].set_axis_off()

    for ax, (name, model) in zip(axes[1:], models.items()):

        # Prepare features
        features = model.inversible_forward(Variable(images).cuda())
        # z = features.view(2, 1, 32, 32)
        # imshow(torchvision.utils.make_grid(z.data.cpu(), 16))
        # plt.savefig(f'features{label}_{i}.png')
        # plt.close()

        interpolation = torch.stack(
            [t*features.data[-1]+(1-t)*features.data[0] for t in steps],
            dim=0
        )

        generated = model.generate(interpolation.cuda())
        score = torch.norm(torch.nn.functional.pad(images, (2,2,2,2)) - generated.cpu(), 2)
        print(f"{name:<30}: {score: >8.2f}")

        img = torchvision.utils.make_grid(generated.data.cpu(), 16)
        img = (img - img.min())/(img.max()-img.min())
        npimg = img.numpy()

        ax.imshow(np.transpose(npimg, (1, 2, 0)))
        ax.text(-7, 17, name, horizontalalignment='right')
        ax.set_axis_off()
    fig.savefig(f'figures/aug_interpolation.png')
    plt.close(fig)


def compute_values(epoch_short, epoch_long):
    models = load_checkpoints(epoch_short, epoch_long)

    # dataloader = load_augmented(8*16, 16, num_workers=2)
    dataloader = load_mnist(1, shuffle=True, train=False, num_workers=2)
    dataiter = iter(dataloader)
    T = np.linspace(-0,5, 0, 16)
    steps = np.linspace(0, 1, 16)

    scores = pd.DataFrame(columns=models.keys())

    for i in tqdm(range(100)):
        image, label = dataiter.next()

        def generate_interpolation():
            for t in T:
                yield shear(image, t, 0)

        images = torch.stack(list(generate_interpolation()),
                             dim=0)
        images = images.view(16, 1, 28, 28)

        for name, model in models.items():
            # Prepare features
            features = model.inversible_forward(Variable(images).cuda())
            # z = features.view(2, 1, 32, 32)
            # imshow(torchvision.utils.make_grid(z.data.cpu(), 16))
            # plt.savefig(f'features{label}_{i}.png')
            # plt.close()

            interpolation = torch.stack(
                [t*features.data[-1]+(1-t)*features.data[0] for t in steps],
                dim=0
            )

            generated = model.generate(interpolation.cuda())
            score = torch.norm(torch.nn.functional.pad(images, (2,2,2,2)) - generated.cpu(), 2)
            scores = scores.append({name: score}, ignore_index=True)

    print(scores.mean())
    print(scores.std())


if __name__ == "__main__":
    fire.Fire()
