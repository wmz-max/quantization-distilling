import os
from datetime import datetime
import argparse
from argparse import ArgumentParser

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from modules import CIFAR10_Module
from utils.callbacks import GetData


def get_callbacks(hparams):
    callbacks = [LearningRateMonitor()]
    if hparams.get_data:
        callbacks.append(GetData(**vars(hparams)))
    return callbacks


def get_datamodule(hparams):
    return {
        'cifar10': CIFAR10_Module,
    }[hparams.dataset]  # 若是cifar10就返回CIFAR10_Module


def main(hparams):
    do_main(hparams)


def do_main(hparams):

    # Model
    data_module = get_datamodule(hparams)
    litmodel = data_module(hparams)  # 去modules里找执行哪个数据.py

    # Trainer
    log_name = '_'.join(filter(None, [  # Remove empty string by filtering
        f'{datetime.now().strftime("%m%d_%H%M")}',
        f'{hparams.dataset}',
        f'{hparams.model}',
        'weightquant' if hparams.weight_quant else '',
        f'act_quant_{hparams.act_quant_coef}' if hparams.act_quant_bit else '',
    ]))

    callbacks = get_callbacks(hparams)

    logger = TensorBoardLogger(hparams.log_dir, name=log_name)

    checkpoint = ModelCheckpoint(save_last=hparams.admm_prune)

    kwargs = {}
    if hparams.world_size > 1:
        kwargs['accelerator'] = 'ddp'

    trainer = Trainer(callbacks=callbacks, gpus=hparams.gpus, max_epochs=hparams.max_epochs,
                      deterministic=True, logger=logger,
                      check_val_every_n_epoch=hparams.val_per_n,
                      checkpoint_callback=checkpoint,
                      **kwargs,
                      # limit_train_batches=10,
                      # limit_val_batches=10,
                      )

    # Load best checkpoint
    if hparams.checkpoint is not None:
        if hparams.load_last:
            load_path = os.path.join(hparams.checkpoint, 'last.ckpt')
        else:
            load_path = os.path.join(
                hparams.checkpoint, os.listdir(hparams.checkpoint)[0])
        print(f'Loading {load_path} ...')
        
        litmodel.load_state_dict(torch.load(load_path)['state_dict'])
        print(torch.load(load_path)['state_dict'])
        

    # Test model
    if hparams.test:
        trainer.test(litmodel)
    else:
        trainer.fit(litmodel)

    checkpoint = os.path.join(
        hparams.log_dir, log_name, f'version_{litmodel.logger.version}', 'checkpoints')
    print(f'Saved checkpoint at {checkpoint}')
    return checkpoint


if __name__ == '__main__':
    parser = ArgumentParser(description='aaa',
                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Dataset
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'imagenet', 'mnist'])
    parser.add_argument('--num_class', type=int, default=10)
    parser.add_argument('--train_dir', type=str, default='data')
    parser.add_argument('--val_dir', type=str, default=None,
                        help='Use train_dir if None')

    # General
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--optimizer', type=str, default='sgd',
                        choices=['sgd', 'adam'])
    parser.add_argument('--scheduler', type=str, default='one-cycle',
                        choices=['one-cycle', 'multi-step'])
    parser.add_argument('--schedule', type=int, nargs='+', default=[30, 60])

    # Environment
    parser.add_argument('--gpus', type=str, default='0,')
    parser.add_argument('--workers', type=int, default=4)

    # Training hparams
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    # Visualization and logs
    parser.add_argument('--log_dir', type=str, default='logs/')
    parser.add_argument('--tmp', action='store_true',
                        help='Save checkpoint at /tmp (keep workspace clean)')
    parser.add_argument('--val_per_n', type=int, default=1,
                        help='Do validation every n epochs')
    parser.add_argument('--checkpoint', type=str, default=None)

    # Quantization
    # Weight quantization #
    parser.add_argument('--weight_quant', action='store_true')
    parser.add_argument('--weight_quant_type', type=str, default='int4',
                        choices=['int4', 'single-pow2', 'mix-pow2', 'ex-mix-pow2', '8bit'])

    # Activation quantization #
    parser.add_argument('--act_quant_bit', type=int, default=None,
                        help='Number of bit after act')
    parser.add_argument('--act_quant_coef', type=float, default=0.01)

    # Pruning
    parser.add_argument('--admm_prune', action='store_true',
                        help='Whether to do admm pruning')
    parser.add_argument('--hard_prune', action='store_true',
                        help='Whether to do hard pruning')
    parser.add_argument('--load_last', action='store_true',
                        help='Load last.ckpt (Needed when loading admm pruning checkpoint)')
    parser.add_argument('--prune_method', type=str, default='filter',
                        choices=['kernel', 'channel', 'filter', 'pattern'])
    parser.add_argument('--non_zero', type=int, default=1,
                        help='For pattern pruning (Temporarily support 1-pat only)')
    parser.add_argument('--groups', type=int, default=8,
                        help='For kernel pruning')
    parser.add_argument('--ratio', type=float, default=0.75,
                        help='For filter, channel, kernel pruning')
    parser.add_argument('--admm_epoch', type=int, default=1,
                        help='Compute new admm mask every n epochs')

    # Get data
    parser.add_argument('--get_data', action='store_true',
                        help='Whether to get data for IC testing')
    parser.add_argument('--one_batch', action='store_true',
                        help='Exit program after 1 batchs (Needed when get_data for one batch)')
    parser.add_argument('--output_dir', type=str,
                        default='output/', help='Output dir for get_data')

    # Reproduce
    parser.add_argument('--seed', type=int, default=0,
                        help='For reproducibility')

    parser.add_argument('--test', action='store_true', help='Do test only')

    args = parser.parse_args()  # 必写这步

    # About dataset
    if args.val_dir is None:
        args.val_dir = args.train_dir

    args.act_quant = None


    args.world_size = len(list(filter(None, args.gpus.split(','))))
    args.batch_size //= args.world_size

    main(args)
