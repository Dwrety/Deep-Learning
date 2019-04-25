import torch
import torchvision
import torch.nn as nn
from collections import OrderedDict
from MyLoss import AngleLoss, CenterLoss, AgentCenterLoss, AngleLinear


"""
This code provides a basic DenseNet architecture implementation of the 11-785 HW2P2 face classification
and verification. The input images are 32x32 gray-scale images. The DenseNet captures features from each 
Convolution layers and concatenate them all together.
"""

class _MyDenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_MyDenseLayer, self).__init__()
        # The two consequtive convolutions operations only expand the input 
        # images in filter bank size, the image size is not changed
        self.add_module('norm1', nn.BatchNorm2d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False))
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_MyDenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _MyDenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_MyDenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _MyDenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _MyTransition(nn.Sequential):
    '''
    Make the network match the input size 
    '''
    def __init__(self, num_input_features, num_output_features):
        super(_MyTransition, self).__init__()
        self.add_module('norm_t', nn.BatchNorm2d(num_input_features))
        self.add_module('relu_t', nn.ReLU(inplace=True))
        self.add_module('conv_t', nn.Conv2d(num_input_features, num_output_features, 
                                            kernel_size=1, stride=1, bias=False))
        self.add_module('pool_t', nn.AvgPool2d(kernel_size=2, stride=2))


class MyDenseNet(nn.Module):
    def __init__(self, growth_rate=16, block_config=(4,8,12),
                 num_init_features=32, bn_size=4, drop_rate=0, num_classes=4300):
        super(MyDenseNet, self).__init__()

        # Initial Convolution Block
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=2, stride=2, padding=0))]))

        # Construction Network Iteratively
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _MyDenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i+1), block)
            num_features = num_features + num_layers * growth_rate
            # Add in transition to make dimension match
            if not i == len(block_config) - 1:
                transition_layer = _MyTransition(num_input_features=num_features, num_output_features=num_features//2)
                self.features.add_module('transition%d' % (i+1), transition_layer)
                num_features = num_features//2

        # Final FC Embedding
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.fc1 = nn.Linear(num_features, num_classes)
        # self.fc2 = AngleLinear(num_features, num_classes)

        # Weight Initializations for filters and FC layer 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        features = torch.nn.functional.relu(features, inplace=True)
        features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1)).view(features.size(0), -1)
        out = self.fc1(features)
        # out = self.fc2(out)
        return features, out
    

def MyDenseNet161(**kwargs):
    model = MyDenseNet(num_init_features=96, growth_rate=48, block_config=(6, 12, 36, 24),**kwargs)
    return model


if __name__ == '__main__':
    num_class = 2300
    model = MyDenseNet161()
    print(model)
    x = torch.rand(1,3,32,32)
    out = model(x)
    print(out.size())

    import matplotlib.pyplot as plt
    import numpy as np 
    sample = plt.imread("test_classification/medium/25.jpg")
    plt.imshow(sample)
    plt.show()
    sample = sample.transpose(2, 0, 1)
    print(sample.shape)

