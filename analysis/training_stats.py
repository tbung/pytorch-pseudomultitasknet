import json

import fire

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from pseudomultitasknet import PseudoMultiTaskNet


def load_record(path):
    with open(path) as f:
        obj = json.load(f)
    return obj


def plot_record(key):
    basepath = Path.home() / "Projects" / "pytorch-pseudomultitasknet" / "runs"
    record_paths = basepath.glob("**/records.json")

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for path in record_paths:
        obj = load_record(path)
        data = pd.DataFrame.from_records(obj[key],
                                         columns=["time", "step", "value"])

        ax.plot(data.step, data.value, label=obj["name"])
        ax.set_ylim((0.9, 1.01))
        ax.legend()

    fig.savefig(f"{key.replace('/', '_')}.png")


if __name__ == "__main__":
    fire.Fire()
