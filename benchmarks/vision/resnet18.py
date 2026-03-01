"""
ResNet-18 with swappable activation function.

Based on the standard torchvision ResNet but constructed manually so
that every nn.ReLU can be replaced with an arbitrary activation module.
Designed for CIFAR (32×32 inputs) with the common modification of
using a 3×3 first conv (no max-pool), following the practice from
https://github.com/kuangliu/pytorch-cifar.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride, act_fn):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = copy.deepcopy(act_fn)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = copy.deepcopy(act_fn)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.act2(out)
        return out


class ResNet(nn.Module):
    """ResNet with configurable activation and block counts."""

    def __init__(self, block, num_blocks, act_fn, num_classes=10):
        super().__init__()
        self.in_planes = 64
        self._act_fn = act_fn

        # CIFAR-style stem: 3×3 conv, no max-pool
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.act0 = copy.deepcopy(act_fn)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s, self._act_fn))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.act0(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet18(act_fn: nn.Module, num_classes: int = 100) -> ResNet:
    """Construct a ResNet-18 with the given activation function."""
    return ResNet(BasicBlock, [2, 2, 2, 2], act_fn, num_classes=num_classes)
