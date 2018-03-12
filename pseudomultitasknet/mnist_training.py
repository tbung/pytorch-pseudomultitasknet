from pathlib import Path
from datetime import datetime

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

from ignite.trainer import Trainer
from ignite.evaluator import create_supervised_evaluator
from ignite.engine import Events
# from ignite.handlers.logging import log_training_simple_moving_average
from ignite._utils import to_variable

from pseudomultitasknet import PseudoMultiTaskNet
from pseudomultitasknet.data import load_mnist
from pseudomultitasknet.experiment import Experiment, RegisterExperiment

CUDA = torch.cuda.is_available()

base_path = Path('saves')

def create_supervised_trainer(model, optimizer, loss_fn, cuda=False):
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
        loss = loss_fn(outputs, y)
        loss.backward()
        optimizer.step()
        y_pred = torch.log((outputs[0]+outputs[1]+outputs[2])/3)
        return loss.data.cpu().item(), y_pred, y

    return Trainer(_update)

def get_accuracy(history, window_size, transform=lambda x: x):
    y_preds, ys = zip(*map(transform, history[-window_size:]))
    y_pred = torch.cat(y_preds, dim=0)
    y = torch.cat(ys, dim=0)
    indices = torch.max(y_pred, 1)[1]
    correct = torch.eq(indices, y)
    return torch.mean(correct.type(torch.FloatTensor)).numpy()

def get_loss(history, window_size, transform=lambda x: x):
    loss = list(map(transform, history[-window_size:]))
    return np.mean(loss)

def save_checkpoint(model, exp_id):
    path = base_path / exp_id / "checkpoints"
    path.mkdir(parents=True, exist_ok=True)
    path /= "{0}_{1:%Y-%m-%d}_{1:%H-%M-%S}.dat".format(model.name,
                                                             datetime.now())
    torch.save(model.state_dict(), path)

@RegisterExperiment("mnist_base")
class MNISTTraining(Experiment):
    def __init__(self):
        super(MNISTTraining, self).__init__()

        self.name =  "mnist_base"
        self.model = PseudoMultiTaskNet()
        self.batch_size = 256
        self.epochs = 10
        self.lr = 0.1
        self.momentum = 0.9
        self.weight_decay = 1e-5
        self.step_size = 3
        self.gamma = 0.1
        self.history = []

        def loss(outputs, labels):
            likelihoods = torch.log((outputs[0]+outputs[1]+outputs[2])/3)
            disagreement = torch.sum(torch.abs(outputs[0] - outputs[1]))
            sparsity = torch.norm(outputs[3], 1)

            return F.nll_loss(likelihoods, labels) + 0.01*disagreement + 0.0001*sparsity

        if CUDA: self.model.cuda(0)

        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=10*self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )

        self.scheduler = StepLR(self.optimizer, step_size=self.step_size,
                                gamma=self.gamma)

        self.evaluator = create_supervised_evaluator(self.model, cuda=CUDA)

        self.trainloader = load_mnist(self.batch_size, train=True)
        self.valloader = load_mnist(self.batch_size, train=False)

        self.trainer = create_supervised_trainer(self.model, self.optimizer, loss,
                                                 cuda=CUDA)

    def on_epoch_started(self, state, trainer):
        self.scheduler.step()
        self.progress = tqdm(total=len(self.trainloader))
        del self.history[:]

    def on_epoch_completed(self, state, trainer):
        # self.evaluator.run(self.valloader)
        save_checkpoint(self.model, self.name)
        self.progress.close()

    def on_iteration_completed(self, _, state):
        self.model.free()
        self.progress.update()
        self.history.append(state.output)
        acc = get_accuracy(self.history, self.batch_size, transform=lambda x: x[-2:])
        self.progress.set_postfix(
            acc=f"{acc*100:.3}%"
        )

    def register_hooks(self):
        self.register_trainer(self.trainer, self.trainloader, self.epochs)

        self.trainer.add_event_handler(
            Events.EPOCH_STARTED,
            self.on_epoch_started
        )

        self.trainer.add_event_handler(
            Events.EPOCH_COMPLETED,
            self.on_epoch_completed
        )

        self.trainer.add_event_handler(
            Events.ITERATION_COMPLETED,
            self.on_iteration_completed
        )


if __name__ == "__main__":
    experiment = MNISTTraining()
    experiment.run()
