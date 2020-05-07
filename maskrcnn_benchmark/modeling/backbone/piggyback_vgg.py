"""Contains various network definitions."""
from __future__ import division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models

from . import piggyback_layers as nl





class View(nn.Module):
    """Changes view using a nn.Module."""

    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)


class ModifiedVGG16(nn.Module):
    """VGG16 with support for multiple classifiers."""

    def __init__(self, cfg):
        super(ModifiedVGG16, self).__init__()
        
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
            vgg = models.vgg16(pretrained=True)
            print('Creating model: No mask layers.')
        else:
            # Get the one with masks and pretrained model.
            vgg = vgg16(mask_init, mask_scale, threshold_fn)
            vgg16_pretrained = models.vgg16(pretrained=True)
            # Copy weights from the pretrained to the modified model.
            for module, module_pretrained in zip(vgg.modules(), vgg16_pretrained.modules()):
                #print('module\n',module)
                #print('module_pretrained\n',module_pretrained)
                if 'ElementWise' in str(type(module)):
                    module.weight.data.copy_(module_pretrained.weight.data)
                    module.bias.data.copy_(module_pretrained.bias.data)
            print('Creating model: Mask layers created.')
        '''
        print('ori vgg features\n',nn.Sequential(*list(vgg.features._modules.values())))
        print('RCNN_base\n',nn.Sequential(*list(vgg.features._modules.values())[:-1]))
        exit()
        '''
        # Create the feature generator.
        for name, module in vgg.features.named_children():
            if name != '30':
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


class VGG(nn.Module):

    def __init__(self, features, mask_init, mask_scale, threshold_fn, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.classifier = nn.Sequential(
            nl.ElementWiseLinear(
                512 * 7 * 7, 4096, mask_init=mask_init, mask_scale=mask_scale,
                threshold_fn=threshold_fn),
            nn.ReLU(True),
            nn.Dropout(),
            nl.ElementWiseLinear(
                4096, 4096, mask_init=mask_init, mask_scale=mask_scale,
                threshold_fn=threshold_fn),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x




def make_layers(cfg, mask_init, mask_scale, threshold_fn, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nl.ElementWiseConv2d(
                in_channels, v, kernel_size=3, padding=1,
                mask_init=mask_init, mask_scale=mask_scale,
                threshold_fn=threshold_fn)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg16(mask_init='1s', mask_scale=1e-2, threshold_fn='binarizer', **kwargs):
    """VGG 16-layer model (configuration "D")."""
    model = VGG(make_layers(cfg['D'], mask_init, mask_scale, threshold_fn),
                mask_init, mask_scale, threshold_fn, **kwargs)
    return model


