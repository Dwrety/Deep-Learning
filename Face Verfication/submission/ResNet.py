import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from MyLoss import AngleLoss, CenterLoss, AgentCenterLoss, AngleLinear


def conv3x3(in_channels, out_channels, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_channels, out_channels, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)



class _ResidualBlock(nn.Module):
    expansion = 1 

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(_ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels * self.expansion)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class _WideBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)
        return out


class _BottleNeckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(_BottleNeckBlock, self).__init__()
        self.conv1 = conv1x1(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels, stride)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(p=0.5)
        self.conv3 = conv1x1(out_channels, out_channels * self.expansion)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=2300, zero_init_residual=False):
        super(ResNet, self).__init__()

        self.in_channels = 64
        # Initial Convolution
        self.features = nn.Sequential(OrderedDict([('conv1', nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,bias=False)),
                                ('norm1',nn.BatchNorm2d(64)),
                                ('relu1', nn.ReLU(inplace=True)),
                                ('pool1',nn.MaxPool2d(kernel_size=2, stride=2, padding=0))]))
        
        self.features.add_module('layer1',self._make_layer(block, 64, layers[0]))
        self.features.add_module('layer2',self._make_layer(block, 128, layers[1], stride=2))
        self.features.add_module('layer3',self._make_layer(block, 256, layers[2], stride=2))
        self.features.add_module('layer4',self._make_layer(block, 512, layers[3], stride=2))
        self.features.add_module('avgpool',nn.AdaptiveAvgPool2d((1, 1)))
        self.fc1 = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != channels * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_channels, channels * block.expansion, stride),
                nn.BatchNorm2d(channels * block.expansion),
            )
        layers = []
        layers.append(('block0', block(self.in_channels, channels, stride, downsample)))
        self.in_channels = channels * block.expansion
        i = 1
        for _ in range(1, blocks):
            layers.append(('block{}'.format(i), block(self.in_channels, channels)))
        i += 1
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


def easyresnet(**kwargs):
    model = ResNet(_BottleNeckBlock, [3, 4, 6, 3], **kwargs)
    return model 



# num_classes = 2300
# model = easyresnet(num_classes=num_classes)
# x = torch.rand(1,3,32,32)
# print(model)
# x = model.features(x)
# print(x.size())

