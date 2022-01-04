import torch
import torch.nn as nn
from torchvision.models.resnet import (
    BasicBlock as _BasicBlock,
    ResNet as _ResNet,
    model_urls,
    load_state_dict_from_url,
)
from utils.quant_layer import QuantLayer


class BasicBlock(_BasicBlock):
    def __init__(self, *args, **kwargs):
        super(BasicBlock, self).__init__(*args, **kwargs)
        self.quant1 = QuantLayer()
        self.quant2 = QuantLayer()
        self.quant3 = QuantLayer()
        self.quant_shortcut = QuantLayer()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.quant1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.quant2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
            identity = self.quant_shortcut(identity)

        out += identity
        out = self.relu(out)
        out = self.quant3(out)

        return out


class ResNet(_ResNet):

    def __init__(self, *args, **kwargs):
        super(ResNet, self).__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.quant1 = QuantLayer()
        self.quant_avg = QuantLayer()
        self.quant_fc = QuantLayer()

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.quant1(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.quant_avg(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.quant_fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)