from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable

from ignite.trainer import Trainer
from ignite.engine import Events

from pseudomultitasknet.data import load_augmented
from pseudomultitasknet.experiment import RegisterExperiment

from pseudomultitasknet.mnist_training import MNISTTraining, get_loss

CUDA = torch.cuda.is_available()

class InterpolationTraining(MNISTTraining):
    def __init__(self):
        super(InterpolationTraining, self).__init__()

        self.aug_batch_size = 16
        self.aug_lr = 1e-4

        self.aug_optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.aug_lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )

        self.augloader = load_augmented(self.aug_batch_size)

        self.aug_trainer = Trainer(self.update)

        def on_aug_epoch_started(trainer):
            self.progress = tqdm(total=len(self.augloader))

        def on_aug_epoch_completed(trainer):
            self.progress.close()

        def on_aug_iteration_completed(trainer):
            self.progress.update()
            loss = get_loss(trainer, self.aug_batch_size)
            self.progress.set_postfix(
                loss=f"{loss:.3}"
            )

        self.aug_trainer.add_event_handler(
            Events.EPOCH_STARTED,
            on_aug_epoch_started
        )

        self.aug_trainer.add_event_handler(
            Events.EPOCH_COMPLETED,
            on_aug_epoch_completed
        )

        self.aug_trainer.add_event_handler(
            Events.ITERATION_COMPLETED,
            on_aug_iteration_completed
        )

    def on_epoch_completed(self, trainer):
        super(InterpolationTraining, self).on_epoch_completed(trainer)
        self.aug_trainer.run(self.augloader)

    def interpolation(self, outputs):
        steps = np.linspace(0, 1, 16)
        return torch.stack([(1-t)*outputs.data[0]+t*outputs.data[1] for t in steps], dim=0)

    def update(self, inputs):
        if CUDA:
            inputs = Variable(inputs).cuda()

        outputs = self.model.inversible_forward(
            torch.stack((inputs[0], inputs[-1]), dim=0)
        )

        interpolation = self.interpolation(outputs)

        generated = self.model.generate(Variable(interpolation.cuda()))

        self.aug_optimizer.zero_grad()

        loss = F.l1_loss(F.pad(inputs, (2,2,2,2)), generated)

        loss.backward()

        # Free the memory used to store activations
        self.model.free()

        self.aug_optimizer.step()

        return loss.cpu().data[0]

if __name__ == "__main__":
    experiment = InterpolationTraining()
    experiment.run()
