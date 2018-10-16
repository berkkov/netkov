import torch
import torch.nn as nn
from InceptionNet import BasicConv2d
import NetworkUtils as utils


class LargeShallowNet(nn.Module):
    def __init__(self):
        super(LargeShallowNet, self).__init__()

        self.subsample = nn.AvgPool2d(kernel_size=4, stride=4, padding=0, count_include_pad=False)
        self.conv1 = BasicConv2d(in_channels=3, out_channels=96, kernel_size=8, stride=4, padding=4)
        # self.conv2 = BasicConv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=0)
        self.maxpool = nn.MaxPool2d(kernel_size=7, stride=4, padding=3)

    def forward(self, x):
        x = self.subsample(x)
        x = self.conv1(x)
        # x = self.conv2(x)
        x = self.maxpool(x)
        return x


class SmallShallowNet(nn.Module):
    def __init__(self):
        super(SmallShallowNet, self).__init__()

        self.subsample = nn.AvgPool2d(kernel_size=8, stride=8, padding=0, count_include_pad=False)
        self.conv1 = BasicConv2d(in_channels=3, out_channels=96, kernel_size=8, stride=4, padding=4)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):

        x = self.subsample(x)
        x = self.conv1(x)
        x = self.maxpool(x)
        return x


class ShallowNet(nn.Module):
    def __init__(self):
        super(ShallowNet, self).__init__()
        self.LargeShallow = LargeShallowNet()
        self.SmallShallow = SmallShallowNet()
        self.flatten = utils.Flatten()

    def forward(self, x):

        xS = self.SmallShallow(x)
        xS = self.flatten(xS)

        xL = self.LargeShallow(x)
        xL = self.flatten(xL)

        x = torch.cat((xS, xL), 1)

        return x
