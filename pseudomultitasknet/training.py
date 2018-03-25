from pathlib import Path
from datetime import datetime
from functools import partial

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable

import numpy as np

from tensorboardX import SummaryWriter

from ignite._utils import to_variable

from tqdm import tqdm

from pseudomultitasknet import PseudoMultiTaskNet
from pseudomultitasknet.data import load_mnist

CUDA = torch.cuda.is_available()

base_path = Path('saves')


def shear(image, value, axis):
    image = image.cpu()
    image = image.squeeze()
    base = image.min()
    coordinates = torch.nonzero(image != base).t().float()

    fill = image[image != base].clone()

    rangey, rangex = image.size()

    half = image.size(axis) // 2
    coordinates[1 - axis] = torch.ceil(coordinates[1 - axis] +
                                       (coordinates[axis]-half)*value)

    if coordinates.max() < rangex and coordinates.min() >= 0:
        result = torch.sparse.FloatTensor(coordinates.long(), fill,
                                          image.size()).to_dense()
        result[result == 0] = base
        return result
    else:
        # print('Overflow')
        return image


def interpolate(outputs):
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
    x = to_variable(x, cuda=CUDA)
    y = to_variable(y, cuda=CUDA)
    return x, y


def process_batch_classification(x, y, model, loss_fn, optimizer):
    optimizer.zero_grad()
    outputs = model(x)
    loss, y_pred = loss_fn(outputs, y)
    loss.backward()
    # Free the memory used to store activations
    model.free()
    optimizer.step()

    correct = y.eq(y_pred).float().sum()

    return loss, correct


def process_batch_interpolation(x, y, model, loss_fn, optimizer):
    # T = np.linspace(-0.5, 0.5, 16)

    # def generate_interpolation():
    #     for t in T:
    #         yield torch.stack(list(map(partial(shear, value=t, axis=1),
    #                                    x.float())), dim=0)

    # aug_data = torch.stack(list(generate_interpolation()),
    #                        dim=0).transpose(0, 1).contiguous()
    # x = aug_data.view(-1, 1, 28, 28).cuda()

    def generate_endpoints():
        for i in range(0, x.size(0), 16):
            yield x[i]
            yield x[i+15]

    outputs = model.inversible_forward(
        torch.stack(list(generate_endpoints()), dim=0)
    )

    interpolation = interpolate(outputs)

    generated = model.generate(Variable(interpolation.cuda()))

    optimizer.zero_grad()

    loss = loss_fn(x, generated)

    loss.backward()

    # Free the memory used to store activations
    model.free()

    optimizer.step()

    return loss.data.cpu().item()


def epoch_classification(writer, epoch, dataloader, model, loss_fn, optimizer,
                         scheduler):
    scheduler.step()
    total_loss = 0
    total_correct = 0
    total = 0

    N = len(dataloader)
    t = tqdm(dataloader, ascii=True, desc='{}'.format(epoch).rjust(2))
    for i, batch in enumerate(t):
        model.train()
        x, y = _prepare_batch(batch)
        loss, correct = process_batch_classification(
            x, y, model, loss_fn, optimizer
        )
        total_loss += loss
        total_correct += correct
        total += y.size(0)
        acc = total_correct / total
        writer.add_scalar("train/loss", loss, epoch*N + i)
        writer.add_scalar("train/acc", acc, epoch*N + i)

        t.set_postfix(
            loss=f"{total_loss/(i+1):.2f}",
            acc=f"{acc:.1%}"
        )

    writer.add_scalar("train/epoch_loss", total_loss, epoch)
    writer.add_scalar("train/epoch_acc", acc, epoch)


def epoch_interpolation(writer, epoch, dataloader, model, aug_loss_fn,
                        aug_optimizer, aug_scheduler):
    aug_scheduler.step()
    total_iloss = 0

    N = len(dataloader)
    t = tqdm(dataloader, ascii=True, desc='{}'.format(epoch).rjust(2))
    for i, batch in enumerate(t):
        model.train()
        x, y = _prepare_batch(batch)
        interpolation_loss = process_batch_interpolation(
            x, y, model, aug_loss_fn, aug_optimizer
        )
        total_iloss += interpolation_loss
        writer.add_scalar("train/interpol_loss", interpolation_loss,
                          epoch*N + i)
        t.set_postfix(
            iloss=f"{total_iloss/(i+1):.2f}",
        )
    writer.add_scalar("train/epoch_interpol_loss", total_iloss, epoch)


def epoch_after_epoch(writer, epoch, dataloader, model, loss_fn, aug_loss_fn,
                      optimizer, aug_optimizer, scheduler, aug_scheduler):
    epoch_classification(writer, epoch, dataloader, model, loss_fn, optimizer,
                         scheduler)
    epoch_interpolation(writer, epoch, dataloader, model, aug_loss_fn,
                        aug_optimizer, aug_scheduler)


def epoch_mixed(writer, epoch, dataloader, model, loss_fn, aug_loss_fn,
                optimizer, aug_optimizer, scheduler, aug_scheduler):
    scheduler.step()
    aug_scheduler.step()
    total_loss = 0
    total_iloss = 0
    total_correct = 0
    total = 0

    N = len(dataloader)
    t = tqdm(dataloader, ascii=True, desc='{}'.format(epoch).rjust(2))
    for i, batch in enumerate(t):
        model.train()
        x, y = _prepare_batch(batch)
        loss, correct = process_batch_classification(
            x, y, model, loss_fn, optimizer
        )
        interpolation_loss = process_batch_interpolation(
            x, y, model, aug_loss_fn, aug_optimizer
        )
        total_loss += loss
        total_iloss += interpolation_loss
        total_correct += correct
        total += y.size(0)
        acc = total_correct / total
        writer.add_scalar("train/loss", loss, epoch*N + i)
        writer.add_scalar("train/interpol_loss", interpolation_loss,
                          epoch*N + i)
        writer.add_scalar("train/acc", acc, epoch*N + i)
        t.set_postfix(
            loss=f"{total_loss/(i+1):.2f}",
            iloss=f"{total_iloss/(i+1):.2f}",
            acc=f"{acc:.1%}"
        )

    writer.add_scalar("train/epoch_loss", total_loss, epoch)
    writer.add_scalar("train/epoch_interpol_loss", total_iloss, epoch)
    writer.add_scalar("train/epoch_acc", acc, epoch)


def validate(model, dataloader, pred_fn):
    correct = 0
    total = 0

    model.eval()

    for batch in dataloader:
        images, labels = batch
        if CUDA:
            images, labels = images.cuda(), labels.cuda()
        outputs = model(Variable(images))
        y_pred = pred_fn(outputs)

        # Free the memory used to store activations
        model.free()

        total += labels.size(0)
        correct += (y_pred == labels).float().sum()

    acc = correct / total

    return acc


def train(training):
    for epoch in range(training.epochs):
        training.epoch_function(training.writer, epoch, training.train_loader,
                                training.model, **training.epoch_args)
        acc = validate(training.model, training.val_loader, training.pred_fn)
        training.writer.add_scalar("test/acc", acc, epoch)
        print(f"Accuracy: {acc:.1%}")
        save_checkpoint(training, epoch)


def save_checkpoint(training, epoch):
    path = Path(training.writer.file_writer.get_logdir())
    path /= "checkpoints"
    path.mkdir(parents=True, exist_ok=True)
    path /= f"checkpoint_{epoch}.dat"
    torch.save(training.model.state_dict(), path)


class TrainingSoftMax:
    def __init__(self):
        self.name = "class_sp_inter_ba"
        self.model = PseudoMultiTaskNet()
        if CUDA:
            self.model.cuda(0)
        self.writer = SummaryWriter()
        self.epoch_function = epoch_mixed
        self.epoch_args = {}
        self.epochs = 20
        self.lr = 0.1
        self.momentum = 0.9
        self.weight_decay = 1e-5
        self.step_size = self.epochs//3
        self.gamma = 0.1

        def loss(outputs, labels):
            likelihoods = torch.log((outputs[0]+outputs[1]+outputs[2])/3)
            disagreement = torch.sum(torch.abs(outputs[0] - outputs[1]))
            sparsity = torch.norm(outputs[3], 1)
            _, y_pred = torch.max(likelihoods.data, 1)

            return (F.nll_loss(likelihoods, labels)
                    + 0.01*disagreement
                    + 0.0001*sparsity,
                    y_pred)

        self.epoch_args["loss_fn"] = loss
        self.epoch_args["aug_loss_fn"] = lambda x,y: F.l1_loss(F.pad(x, (2, 2, 2, 2)),y)

        self.epoch_args["optimizer"] = optim.SGD(
            self.model.parameters(),
            lr=10*self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )

        self.epoch_args["scheduler"] = StepLR(self.epoch_args["optimizer"],
                                              step_size=self.step_size,
                                              gamma=self.gamma)

        self.aug_batch_size = 16
        self.batch_size = 16*self.aug_batch_size
        self.aug_lr = 0.01

        self.epoch_args["aug_optimizer"] = optim.SGD(
            self.model.parameters(),
            lr=10*self.aug_lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )
        self.epoch_args["aug_scheduler"] = StepLR(
            self.epoch_args["aug_optimizer"],
            step_size=self.epochs//2,
            gamma=self.gamma
        )

        # self.train_loader = load_augmented(self.batch_size,
        #                                    self.aug_batch_size)
        self.train_loader = load_mnist(self.batch_size, train=True)
        self.val_loader = load_mnist(self.batch_size, train=False)

        for key, value in self.__dict__.items():
            self.writer.add_text(f"config/{key}", str(value))

    def pred_fn(self, outputs):
        likelihoods = torch.log((outputs[0]+outputs[1]+outputs[2])/3)
        _, y_pred = torch.max(likelihoods.data, 1)
        return y_pred

    def __call__(self):
        for key, value in self.__dict__.items():
            self.writer.add_text(f"config/{key}", str(value))
        train(self)


class TrainingSMNB(TrainingSoftMax):
    def __init__(self):
        super(TrainingSMNB, self).__init__()
        self.name = "sm_nb"
        self.model = PseudoMultiTaskNet(no_svd=True)
        if CUDA:
            self.model.cuda(0)

        self.lr = 0.01

        self.epoch_args["optimizer"] = optim.SGD(
            self.model.parameters(),
            lr=10*self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )

        self.epoch_args["scheduler"] = StepLR(self.epoch_args["optimizer"],
                                              step_size=self.step_size,
                                              gamma=self.gamma)

        def loss(outputs, labels):
            likelihoods = torch.log((outputs[0]+outputs[1])/2)
            disagreement = torch.sum(torch.abs(outputs[0] - outputs[1]))
            _, y_pred = torch.max(likelihoods.data, 1)

            return (F.nll_loss(likelihoods, labels)
                    + 0.01*disagreement,
                    y_pred)

        self.epoch_args["loss_fn"] = loss
        self.epoch_function = epoch_classification
        del self.epoch_args["aug_loss_fn"]
        del self.epoch_args["aug_optimizer"]
        del self.epoch_args["aug_scheduler"]


class TrainingSMNBNM(TrainingSoftMax):
    def __init__(self):
        super(TrainingSMNBNM, self).__init__()
        self.name = "class"
        self.model = PseudoMultiTaskNet(no_svd=True)
        if CUDA:
            self.model.cuda(0)

        self.lr = 0.01

        self.epoch_args["optimizer"] = optim.SGD(
            self.model.parameters(),
            lr=10*self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )

        self.epoch_args["scheduler"] = StepLR(self.epoch_args["optimizer"],
                                              step_size=self.step_size,
                                              gamma=self.gamma)

        def loss(outputs, labels):
            likelihoods = torch.log((outputs[0]+outputs[1]+outputs[2])/3)
            disagreement = torch.sum(torch.abs(outputs[0] - outputs[1]))
            _, y_pred = torch.max(likelihoods.data, 1)

            return (F.nll_loss(likelihoods, labels)
                    + 0.01*disagreement,
                    y_pred)

        self.epoch_args["loss_fn"] = loss
        self.epoch_function = epoch_classification
        del self.epoch_args["aug_loss_fn"]
        del self.epoch_args["aug_optimizer"]
        del self.epoch_args["aug_scheduler"]


class TrainingClassSP(TrainingSoftMax):
    def __init__(self):
        super(TrainingClassSP, self).__init__()
        self.name = "class_sp"
        self.epoch_function = epoch_classification
        del self.epoch_args["aug_loss_fn"]
        del self.epoch_args["aug_optimizer"]
        del self.epoch_args["aug_scheduler"]


class TrainingClassInter(TrainingSoftMax):
    def __init__(self):
        super(TrainingClassInter, self).__init__()
        self.name = "class_inter"
        self.model = PseudoMultiTaskNet(no_svd=True)
        if CUDA:
            self.model.cuda(0)

        self.epoch_args["optimizer"] = optim.SGD(
            self.model.parameters(),
            lr=10*self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )

        self.epoch_args["scheduler"] = StepLR(self.epoch_args["optimizer"],
                                              step_size=self.step_size,
                                              gamma=self.gamma)

        self.aug_lr = 1e-5

        self.epoch_args["aug_optimizer"] = optim.SGD(
            self.model.parameters(),
            lr=10*self.aug_lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )
        self.epoch_args["aug_scheduler"] = StepLR(
            self.epoch_args["aug_optimizer"],
            step_size=self.epochs//2,
            gamma=self.gamma
        )

        def loss(outputs, labels):
            likelihoods = torch.log((outputs[0]+outputs[1]+outputs[2])/3)
            disagreement = torch.sum(torch.abs(outputs[0] - outputs[1]))
            _, y_pred = torch.max(likelihoods.data, 1)

            return (F.nll_loss(likelihoods, labels)
                    + 0.01*disagreement,
                    y_pred)

        self.epoch_args["loss_fn"] = loss
        self.epoch_function = epoch_after_epoch


class TrainingClassSPInter(TrainingSoftMax):
    def __init__(self):
        super(TrainingClassSPInter, self).__init__()
        self.name = "class_sp_inter"

        self.aug_lr = 1e-5

        self.epoch_args["aug_optimizer"] = optim.SGD(
            self.model.parameters(),
            lr=10*self.aug_lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )
        self.epoch_args["aug_scheduler"] = StepLR(
            self.epoch_args["aug_optimizer"],
            step_size=self.epochs//2,
            gamma=self.gamma
        )

        self.epoch_function = epoch_after_epoch


class TrainingClassInterBA(TrainingSoftMax):
    def __init__(self):
        super(TrainingClassInterBA, self).__init__()
        self.name = "class_inter_ba"
        self.model = PseudoMultiTaskNet(no_svd=True)
        if CUDA:
            self.model.cuda(0)

        self.epoch_args["optimizer"] = optim.SGD(
            self.model.parameters(),
            lr=10*self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )

        self.epoch_args["scheduler"] = StepLR(self.epoch_args["optimizer"],
                                              step_size=self.step_size,
                                              gamma=self.gamma)
        self.aug_lr = 1e-4

        self.epoch_args["aug_optimizer"] = optim.SGD(
            self.model.parameters(),
            lr=10*self.aug_lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )
        self.epoch_args["aug_scheduler"] = StepLR(
            self.epoch_args["aug_optimizer"],
            step_size=self.epochs//2,
            gamma=self.gamma
        )

        def loss(outputs, labels):
            likelihoods = torch.log((outputs[0]+outputs[1]+outputs[2])/3)
            disagreement = torch.sum(torch.abs(outputs[0] - outputs[1]))
            _, y_pred = torch.max(likelihoods.data, 1)

            return (F.nll_loss(likelihoods, labels)
                    + 0.01*disagreement,
                    y_pred)

        self.epoch_args["loss_fn"] = loss


if __name__ == "__main__":
    training = TrainingClassInterBA()
    training()
    path = Path(training.writer.file_writer.get_logdir())
    path /= "records.json"
    training.writer.export_scalars_to_json(path)
    training.writer.close()
