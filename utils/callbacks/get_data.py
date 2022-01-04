import os
import sys
import shutil
from glob import glob

import torch
import torch.nn as nn
from pytorch_lightning.callbacks.base import Callback

from utils.utils import save_data as _save_data
from utils.quant_layer import QuantLayer
import functools


def scaledFcForward(forward, fc_scale, bias):
    @functools.wraps(forward)
    def wrapper(data):
        return forward(data) * fc_scale + bias
    return wrapper


def printActivation(forward, name, save_data, bit, m=None, is_act=False):
    @functools.wraps(forward)
    def wrapper(input):
        if 'input' in name:
            for i in range(input.size(0)):
                save_data(input[i], f'{name}.{i+1}')
        output = forward(input)
        if 'output' in name:
            q = (bit-1) - m.exp
            for i in range(output.size(0)):
                # m.count += 1
                save_data(output[i], f'{name}.{i+1}', is_act=is_act, q=q)
        return output
    return wrapper


def overloadBN(forward, m):
    @functools.wraps(forward)
    def wrapper(input):
        def expand(x):
            return x.view(1, -1, 1, 1)
        return expand(m.bn_k) * input + expand(m.bn_b)
    return wrapper


def get_modules(model, instance):
    return [
        (name, m)
        for name, m in model.named_modules()
        if isinstance(m, instance)
    ]


class GetData(Callback):
    """
    Get data
    """

    def __init__(self, one_batch=True, output_dir='output', **_kwargs):
        self.one_batch = one_batch
        self.output_dir = output_dir
        self.n_batch = 0

    def on_test_batch_end(self, trainer, pl_module, *args, **kwargs):
        # assert len(pl_module.ori_imgs) == pl_module.hparams.batch_size, (
        #     len(pl_module.ori_imgs)
        # )

        # TODO refactor (1 var count down)
        self.n_batch += 1
        if self.n_batch > 12:
            self._save_imgs(pl_module)

        if self.one_batch:
            self._save_imgs(pl_module)

    def _save_imgs(self, pl_module):
        for i, img in enumerate(pl_module.ori_imgs):
            _save_data(img.contiguous(),
                       f'im{i+1}', output_dir=self.output_dir)
        _save_data(pl_module.labels,
                   f'labels', output_dir=self.output_dir)
        sys.exit(0)

    def on_test_epoch_start(self, trainer, pl_module):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir)
        ckpts = glob(f'{pl_module.hparams.checkpoint}/*.ckpt')
        assert len(ckpts) == 1, 'No checkpoint is found'
        shutil.copyfile(ckpts[0], f'{self.output_dir}/{pl_module.hparams.model}.ckpt')

        save_data = functools.partial(_save_data, output_dir=self.output_dir)

        convs = get_modules(pl_module.model, nn.Conv2d)
        bns = get_modules(pl_module.model, nn.BatchNorm2d)
        fcs = get_modules(pl_module.model, nn.Linear)
        quants = get_modules(pl_module.model, QuantLayer)
        assert len(convs) == len(bns), "Each conv must be followed by bn"

        # Convert conv weight to int, then modify bn to recover the change
        for (_, conv), (_, bn) in zip(convs, bns):
            conv.weight.data = (conv.weight.data / conv.scale).round()
            bn.running_mean = bn.running_mean / conv.scale
            bn.weight.data = bn.weight.data * conv.scale

        # Convert fc to int, then modify forward to recover the change
        for fc_name, fc in fcs:
            fc.bias_ = fc.bias.clone()
            fc.bias.data[:] = 0.0
            fc.weight.data = (fc.weight.data / fc.scale).round()
            fc.forward = scaledFcForward(fc.forward, fc.scale, fc.bias_)
            # save_data(torch.tensor(fc.scale), f'{fc_name}.k', to_hex=True)

        # Convert bn to the form of `bn_k * input + bn_b`
        for name, m in bns:
            m.bn_k = m.weight.data / (m.running_var.sqrt()+1e-6)
            m.bn_b = -m.weight.data * m.running_mean / \
                (m.running_var.sqrt()+1e-6) + m.bias.data
            m.bn_k = m.bn_k.half().float()
            m.bn_b = m.bn_b.half().float()
            m.forward = overloadBN(m.forward, m)

        bit = pl_module.hparams.act_quant_bit
        # Print input of every convs and fcs
        for name, m in convs + fcs:
            m.forward = printActivation(m.forward, f'{name}.input', save_data, bit)

        # Print output of every quants
        for name, m in quants:
            m.forward = printActivation(
                m.forward, f'{name}.output', save_data, bit, m=m)

        # For last output, print int
        for name, m in quants[-1:]:
            m.forward = printActivation(
                m.forward, f'{name}.output.int', save_data, bit, m=m, is_act=True)