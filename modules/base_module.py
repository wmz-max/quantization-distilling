from __future__ import print_function

import functools
from argparse import Namespace
import shutil
import os
import sys
import torch.nn.functional as F
import torch
import torchvision
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR, OneCycleLR
from pytorch_lightning.metrics import Accuracy
# import torchmetrics

from utils import QuantOp, activation_quantization
# from models import resnet_cifar
from models import MODELS
from utils.quant_layer import QuantLayer
# from utils import admm
from utils import lazy_property
# from utils.utils import save_data

import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
# from models.lenet import LeNet

from .cifar10_module_teacher import CIFAR10_Module_Teacher


def identity_forward(forward):
    @functools.wraps(forward)
    def wrapper(input):
        return input

    return wrapper


def quant_forward(forward, m, bits):
    @functools.wraps(forward)
    def wrapper(input):
        out = input
        q = bits - m.exp
        return activation_quantization(out, bits=bits, q=q)[0]

    return wrapper


def get_model(hparams, pretrained=False):

    model = MODELS[hparams.dataset][hparams.model](num_classes=hparams.num_class)
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            m.register_buffer(f'scale', torch.tensor(1.0))

        if isinstance(m, QuantLayer):
            if hparams.test and hparams.act_quant_bit:
                m.forward = quant_forward(m.forward, m, bits=hparams.act_quant_bit - 1)  # 减一是因为若要量化为4bit,是-7-+7
    return model


def get_model1(hparams, pretrained=False):
    load_path = 'pth/resnet-cifar10/0102_0803_cifar10_resnet18/version_0/checkpoints/epoch=99-step=78099.ckpt'
    model1 = CIFAR10_Module_Teacher()
    model1.load_state_dict(torch.load(load_path)['state_dict'], strict=False)
    return model1.model


class BaseModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        if isinstance(hparams, dict):  # 如果 hparams是dict类型
            hparams = Namespace(**hparams)

        self.hparams = hparams
        self.criterion = nn.CrossEntropyLoss()
        self.model = get_model(hparams)
        self.model1 = get_model1(hparams)
        self.train_size = len(self.train_dataloader().dataset)
        self.val_size = len(self.val_dataloader().dataset)
        self.acc = Accuracy(compute_on_step=False)
        self.train_acc = Accuracy()
        self.labels = None

    @lazy_property
    def weight_quant_op(self):
        return QuantOp(self.model, quant_conv=self.hparams.weight_quant_type)

    @lazy_property
    def act_quant_alphas(self):
        '''Get all unique alphas. Filtering using id'''
        return list({
                        id(m.alpha): (name, m)
                        for name, m in self.model.named_modules()
                        if isinstance(m, QuantLayer)
                    }.values())

    @lazy_property
    def min_act_quant(self):
        return torch.tensor(2 ** -2, device=self.device)

    @lazy_property
    def admm(self):
        target_modules = {
            name: m
            for name, m in self.model.named_modules()
            if (
                    isinstance(m, nn.Conv2d)
                    and m.weight.size(3) == 3
                    and m.weight.size(1) > 3
            )
        }
        return admm.ADMM(target_modules, self.hparams)

    def on_train_batch_end(self, *args,
                           **kwargs):  # 一个epoch是要跑完成整个数据集，所以一个epoch有很多batch，每次batch就是一次前向传播和一次反向传播，所以一次batch完事就量化一次weight
        if self.hparams.weight_quant:
            self.weight_quant_op.quantization()



    def on_after_backward(self):
        if self.hparams.weight_quant:
            self.weight_quant_op.restore()  # target_module

    def forward(self, batch):
        images, labels = batch
        if self.hparams.get_data:
            if self.labels is None:
                self.labels = labels
            else:
                self.labels = torch.cat([self.labels, labels])
        if self.hparams.act_quant_bit:
            images, q = activation_quantization(
                images, bits=self.hparams.act_quant_bit - 1)
            if self.hparams.get_data:
                save_data(q, f'img.input.q', output_dir=self.hparams.output_dir)

        logits = self.model(images)
        logits1 = self.model1(images)
        ce_loss = self.criterion(logits, labels)
        # print(logits1)
        # print(logits)
        # print(labels)
        loss_distilled = nn.functional.kl_div(F.log_softmax(logits, dim=-1), F.softmax(logits1, dim=-1))
        loss = 0.3 * ce_loss + 0.7 * loss_distilled

        if self.hparams.admm_prune:
            admm_loss = self.admm.get_admm_loss()
            loss += admm_loss

        return loss, ce_loss, logits.argmax(dim=1), labels

    def training_step(self, batch, batch_nb):
        loss, _, preds, labels = self(batch)

        acc = self.train_acc(preds, labels)

        if self.hparams.act_quant_bit and self.current_epoch != 0:
            for _, m in self.act_quant_alphas:
                act_quant_loss = torch.max(m.alpha, self.min_act_quant)
                loss += self.hparams.act_quant_coef * act_quant_loss

        logs = {
            'loss/train': loss,
            'accuracy/train': acc,
            **{
                f'q/{name}': m.alpha
                for name, m in self.act_quant_alphas
            },
        }

        self.log_dict(logs)
        return loss

    def validation_step(self, batch, batch_nb):
        _, ce_loss, preds, labels = self(batch)
        self.acc.update(preds, labels)
        self.log('val_loss', ce_loss, prog_bar=True)
        self.log('val_acc', self.acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log('checkpoint_on', self.acc, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate,
                                    weight_decay=self.hparams.weight_decay, momentum=0.9)

        steps_per_epoch = (
                                  self.train_size // self.hparams.batch_size) // self.hparams.world_size

        if self.hparams.scheduler == 'one-cycle':
            scheduler = OneCycleLR(optimizer, max_lr=self.hparams.learning_rate,
                                   steps_per_epoch=steps_per_epoch,
                                   epochs=self.hparams.max_epochs)
        elif self.hparams.scheduler == 'multi-step':
            scheduler = MultiStepLR(optimizer, milestones=[
                e * steps_per_epoch for e in self.hparams.schedule])

        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'name': 'learning_rate',
        }
        return [optimizer], [scheduler]

    def train_dataloader(self):
        dataset = self.dataset(root=self.hparams.train_dir, train=True,
                               transform=self.train_transform(), download=True)
        dataloader = DataLoader(dataset, batch_size=self.hparams.batch_size,
                                num_workers=self.hparams.workers, shuffle=True, drop_last=True, pin_memory=True)
        return dataloader

    def val_dataloader(self):
        dataset = self.dataset(root=self.hparams.val_dir,
                               train=False, transform=self.val_transform())
        if self.hparams.get_data:
            dataloader = DataLoader(dataset, batch_size=self.hparams.batch_size,
                                    num_workers=1,
                                    # shuffle=False, pin_memory=True)
                                    shuffle=False)
        else:
            dataloader = DataLoader(dataset, batch_size=self.hparams.batch_size,
                                    num_workers=self.hparams.workers, shuffle=False, pin_memory=True)
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()

