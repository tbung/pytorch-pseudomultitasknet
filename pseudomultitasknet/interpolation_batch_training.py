from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable

from ignite.trainer import Trainer
from ignite.engine import Events
from ignite._utils import to_variable

from pseudomultitasknet.data import load_augmented
from pseudomultitasknet.experiment import RegisterExperiment

from pseudomultitasknet.mnist_training import MNISTTraining, get_loss

CUDA = torch.cuda.is_available()

class InterpolationBatchTraining(MNISTTraining):
    def __init__(self):
        super(InterpolationBatchTraining, self).__init__()

        self.aug_batch_size = 16
        self.batch_size = 8*self.aug_batch_size
        self.aug_lr = 1e-4

        self.aug_optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.aug_lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )

        self.augloader = load_augmented(self.batch_size, self.aug_batch_size)

        def loss(outputs, labels):
            likelihoods = torch.log((outputs[0]+outputs[1]+outputs[2])/3)
            disagreement = torch.sum(torch.abs(outputs[0] - outputs[1]))
            sparsity = torch.norm(outputs[3], 1)

            return F.nll_loss(likelihoods, labels) + 0.01*disagreement + 0.0001*sparsity

        self.trainer = create_supervised_trainer(self.model, self.optimizer,
                                                 self.aug_optimizer,
                                                 loss, cuda=True)


def create_supervised_trainer(model, optimizer, aug_optimizer, loss_fn, cuda=False):
    """
    Factory function for creating a trainer for supervised models
    Args:
        model (torch.nn.Module): the model to train
        optimizer (torch.optim.Optimizer): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        cuda (bool, optional): whether or not to transfer batch to GPU (default: False)
    Returns:
        Trainer: a trainer instance with supervised update function
    """
    def _interpolation(outputs):
        steps = np.linspace(0, 1, 16)

        def generate_interpolation():
            for i in range(0, outputs.size(0), 2):
                for t in steps:
                    yield (1-t)*outputs.data[i]+t*outputs.data[i+1]

        return torch.stack(
            list(generate_interpolation()),
            dim=0
        )

    def _prepare_batch(batch):
        x, y = batch
        x = to_variable(x, cuda=cuda)
        y = to_variable(y, cuda=cuda)
        return x, y

    def _update(batch):
        model.train()
        optimizer.zero_grad()
        x, y = _prepare_batch(batch)
        outputs = model(x)
        loss_forw = loss_fn(outputs, y)
        loss_forw.backward()
        optimizer.step()
        y_pred = torch.log((outputs[0]+outputs[1]+outputs[2])/3)

        def generate_endpoints():
            for i in range(0, x.size(0), 16):
                yield x[i]
                yield x[i+15]

        outputs = model.inversible_forward(
            torch.stack(list(generate_endpoints()), dim=0)
        )

        interpolation = _interpolation(outputs)

        generated = model.generate(Variable(interpolation.cuda()))

        aug_optimizer.zero_grad()

        loss_back = F.l1_loss(F.pad(x, (2,2,2,2)), generated)

        loss_back.backward()

        # Free the memory used to store activations
        model.free()

        aug_optimizer.step()

        return (loss_back.data.cpu().item(), loss_forw.data.cpu().item(),
                y_pred, y)

    return Trainer(_update)


if __name__ == "__main__":
    experiment = InterpolationBatchTraining()
    experiment.run()
