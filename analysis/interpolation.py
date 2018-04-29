from pseudomultitasknet.data import load_mnist
from pseudomultitasknet import PseudoMultiTaskNet, PseudoMultiTaskNetMult

import torch

import torchvision

from torch.autograd import Variable

import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from collections import OrderedDict

import fire
import json


epoch_num = 17


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


def generater_plots(epoch_short, epoch_long):
    models = load_checkpoints(epoch_short, epoch_long)

    dataloader = load_mnist(10000, shuffle=False, train=False, num_workers=0)
    dataiter = iter(dataloader)
    images, labels = dataiter.next()

    heights = [images.size(2)] * len(models)

    fig_width = 8.0
    fig_height = fig_width * sum(heights) / (images.size(3) * 16)

    steps = np.linspace(0, 1, 16)

    for label in range(13):
        if label < 10:
            ones = images.index_select(0, torch.nonzero(labels == label).squeeze())
        else:
            ones = images

        for i in range(4):
            idx = torch.LongTensor(2).random_(0, ones.size(0))
            samples = ones.index_select(0, idx)

            fig, axes = plt.subplots(
                len(models), 1,
                figsize=(fig_width + 4, fig_height),
                gridspec_kw={'height_ratios': heights}
            )
            fig.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0,
                                top=1)

            for ax, (name, model) in zip(axes, models.items()):

                # Prepare features
                features = model.inversible_forward(Variable(samples).cuda())
                # z = features.view(2, 1, 32, 32)
                # imshow(torchvision.utils.make_grid(z.data.cpu(), 16))
                # plt.savefig(f'features{label}_{i}.png')
                # plt.close()

                interpolation = torch.stack(
                    [t*features.data[0]+(1-t)*features.data[1] for t in steps],
                    dim=0
                )

                generated = model.generate(interpolation.cuda())

                img = torchvision.utils.make_grid(generated.data.cpu(), 16)
                img = (img - img.min())/(img.max()-img.min())
                npimg = img.numpy()

                ax.imshow(np.transpose(npimg, (1, 2, 0)))
                ax.text(-7, 17, name, horizontalalignment='right')
                ax.set_axis_off()
            fig.savefig(f'figures/new_interpolation{label}_{i}.png')
            plt.close(fig)


if __name__ == "__main__":
    fire.Fire(generater_plots)
