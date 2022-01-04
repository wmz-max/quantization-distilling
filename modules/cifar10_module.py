import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import numpy as np

from .base_module import BaseModule


class CIFAR10_Module(BaseModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ori_imgs = []

    @property
    def mean(self):
        return (0.4914, 0.4822, 0.4465)

    @property
    def std(self):
        return (0.247, 0.243, 0.261)

    def append_ori(self, x):
        x_ = np.transpose(np.array(x), (2, 0, 1))
        x_ = torch.tensor(x_)
        self.ori_imgs.append(x_)
        return x

    def train_dataloader(self):
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),#归一化
                                              transforms.Normalize(self.mean, self.std)])#标准化 均值变为0，标准差变为1
        dataset = CIFAR10(root=self.hparams.train_dir,
                          train=True, download=True, transform=transform_train)
        dataloader = DataLoader(dataset, batch_size=self.hparams.batch_size,
                                num_workers=4, shuffle=True, drop_last=True, pin_memory=True)
        return dataloader

    def val_dataloader(self):#get_data为什么要下载数据的时候有改变为看懂

        if self.hparams.get_data:
            transform_val = transforms.Compose([
                transforms.Lambda(self.append_ori),#Lambda表示调用自己的函数来改变图片
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])
        else:
            transform_val = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(self.mean, self.std)])
        dataset = CIFAR10(root=self.hparams.val_dir,
                          train=False, transform=transform_val)
        # dataloader = DataLoader(
        #     dataset, batch_size=self.hparams.batch_size, num_workers=4, pin_memory=True)

        if self.hparams.get_data:
            dataloader = DataLoader(dataset, batch_size=self.hparams.batch_size,
                                    shuffle=True)
        else:
            dataloader = DataLoader(dataset, batch_size=self.hparams.batch_size,
                                    num_workers=self.hparams.workers, shuffle=False, pin_memory=True)#如果要get_data就多线程并且每个epoch不重新洗牌，pin_memory？为啥要设为true
        return dataloader
