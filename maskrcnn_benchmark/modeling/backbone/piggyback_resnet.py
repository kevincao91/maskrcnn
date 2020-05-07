"""Contains various network definitions."""
from __future__ import division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models
import math

from . import piggyback_layers as nl


class View(nn.Module):
    """Changes view using a nn.Module."""

    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)


class ModifiedResNet(nn.Module):
    """ResNet-50."""

    def __init__(self, cfg):
        super(ModifiedResNet, self).__init__()
        
        self.stages = []
        
        mask_init='1s'
        mask_scale=1e-2
        threshold_fn='binarizer'
        make_model=True
        original=False
        
        if make_model:
            self.make_model(mask_init, mask_scale, threshold_fn, original)

    def make_model(self, mask_init, mask_scale, threshold_fn, original):
        """Creates the model."""
        if original:
            resnet = models.resnet50(pretrained=True)
            print('Creating model: No mask layers.')
        else:
            #print('make_model mask_init', mask_init)
            # Get the one with masks and pretrained model.
            resnet = resnet50(mask_init, mask_scale, threshold_fn)
            '''
            # add by kevin.cao at 20.01.08 ===
            #print(resnet)
            #if self.args.threshold_fn == 'binarizer':
            if True:
                print('Num 0ed out parameters:')
                for idx, module in enumerate(resnet.modules()):
                    if 'ElementWise' in str(type(module)):
                        #print(module.mask_real.data)
                        num_zero = module.mask_real.data.lt(5e-3).sum()
                        total = module.mask_real.data.numel()
                        print(idx, num_zero, total)
            print('-' * 20)
            '''
            # ================================
            resnet_pretrained = models.resnet50(pretrained=True)
            # Copy weights from the pretrained to the modified model.
            for module, module_pretrained in zip(resnet.modules(), resnet_pretrained.modules()):
                if 'ElementWise' in str(type(module)):
                    module.weight.data.copy_(module_pretrained.weight.data)
                    if module.bias:
                        module.bias.data.copy_(module_pretrained.bias.data)
                elif 'BatchNorm' in str(type(module)):
                    module.weight.data.copy_(module_pretrained.weight.data)
                    module.bias.data.copy_(module_pretrained.bias.data)
                    module.running_mean.copy_(module_pretrained.running_mean)
                    module.running_var.copy_(module_pretrained.running_var)
            print('Creating model: Mask layers created.')

        # Create the feature generator.
        for name, module in resnet.named_children():
            if name != 'fc' and name != 'avgpool' and name != 'layer4':
                self.add_module(name, module)
                self.stages.append(name)
        print(self.stages)
        #exit()
    
    def forward(self, x):
        outputs = []
        for stage_name in self.stages:
            x = getattr(self, stage_name)(x)
        outputs.append(x)
        return outputs
    

def conv3x3(in_planes, out_planes, mask_init, mask_scale, threshold_fn, stride=1):
    "3x3 convolution with padding"
    return nl.ElementWiseConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                                padding=1, bias=False, mask_init=mask_init, mask_scale=mask_scale,
                                threshold_fn=threshold_fn)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, mask_init, mask_scale, threshold_fn, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, mask_init,
                             mask_scale, threshold_fn, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, mask_init,
                             threshold_fn, mask_scale)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, mask_init, mask_scale, threshold_fn, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nl.ElementWiseConv2d(
            inplanes, planes, kernel_size=1, bias=False,
            mask_init=mask_init, mask_scale=mask_scale, threshold_fn=threshold_fn)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nl.ElementWiseConv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False,
            mask_init=mask_init, mask_scale=mask_scale, threshold_fn=threshold_fn)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nl.ElementWiseConv2d(
            planes, planes * 4, kernel_size=1, bias=False,
            mask_init=mask_init, mask_scale=mask_scale, threshold_fn=threshold_fn)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, mask_init, mask_scale, threshold_fn, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nl.ElementWiseConv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False,
            mask_init=mask_init, mask_scale=mask_scale, threshold_fn=threshold_fn)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(
            block, 64, layers[0], mask_init, mask_scale, threshold_fn)
        self.layer2 = self._make_layer(
            block, 128, layers[1], mask_init, mask_scale, threshold_fn, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], mask_init, mask_scale, threshold_fn, stride=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], mask_init, mask_scale, threshold_fn, stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nl.ElementWiseConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, mask_init, mask_scale, threshold_fn, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nl.ElementWiseConv2d(
                    self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False,
                    mask_init=mask_init, mask_scale=mask_scale, threshold_fn=threshold_fn),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, mask_init,
                            mask_scale, threshold_fn, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                mask_init, mask_scale, threshold_fn))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet50(mask_init='1s', mask_scale=1e-2, threshold_fn='binarizer', **kwargs):
    """Constructs a ResNet-50 model."""
    #print('resnet50 get in')
    #print('resnet50 mask_init', mask_init)
    model = ResNet(Bottleneck, [3, 4, 6, 3], mask_init,
                   mask_scale, threshold_fn, **kwargs)
    return model
        
        
