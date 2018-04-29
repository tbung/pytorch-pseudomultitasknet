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

from tabulate import tabulate

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
        else:
            model = PseudoMultiTaskNet()
        model.load_state_dict(torch.load(path / f"checkpoint_{epoch}.dat"))
        model.cuda(0)
        models[name] = model

    return models


def write_tables():
    models = load_checkpoints(17, 47)
    d = models["class_sp_inter_ba"].__dict__
    # table = tabulate(d, tablefmt="latex_booktabs")
    # print(table)
    print(d)

if __name__ == "__main__":
    fire.Fire(write_tables)
