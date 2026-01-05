import os
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import numpy as np
import pandas as pd
from torchvision import datasets, transforms

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, dropout_rate=0.0, spect_norm=False):
        super(MLP, self).__init__()
        layers = []
        
        if num_layers == 0:
            final_linear = nn.Linear(input_dim, output_dim)
            layers.append(spectral_norm(final_linear) if spect_norm else final_linear)
        else:
            layers.append(nn.Dropout(p=dropout_rate))
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.LayerNorm(hidden_dim))
            
            for _ in range(num_layers - 2):
                layers.append(nn.Dropout(p=dropout_rate))
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.LayerNorm(hidden_dim))
                
            final_linear = nn.Linear(hidden_dim, output_dim)
            final_linear = spectral_norm(final_linear) if spect_norm else final_linear
            layers.append(final_linear)
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class FEDL(nn.Module):
    def __init__(self, ID_dataset, dropout_rate, device, hidden_dim, num_layers):
        spect_norm = True
        
        super().__init__()
        if ID_dataset == "MNIST":
            num_classes = 10
            backbone = conv_net(dropout_rate, spect_norm)
            f, g_alpha = backbone.convolutions, backbone.linear
            embedding_dim = 576
            
        elif ID_dataset == "CIFAR-10":
            num_classes = 10
            backbone = vgg16(num_classes, dropout_rate, spect_norm)
            f, g_alpha = backbone.features, backbone.classifier
            embedding_dim = 512
            
        elif ID_dataset == "CIFAR-100":
            num_classes = 100
            backbone = resnet18(spect_norm)
            f, g_alpha = backbone.features, backbone.classifier
            embedding_dim = 512
            
        else:
            raise ValueError(f"Unsupported dataset: {ID_dataset}")

        self.f = f
        self.g_alpha = g_alpha
        self.g_tau = MLP(embedding_dim, 1, hidden_dim, num_layers, dropout_rate, spect_norm = False).to(device)
        self.g_p = MLP(embedding_dim, num_classes, hidden_dim, num_layers, dropout_rate, spect_norm = False).to(device)

    def forward(self, x, fix_tau=False, fix_p=None):
        features = self.f(x)
        
        if isinstance(features, (tuple, list)):
            features = features[0]
        if len(features.shape) > 2:
            features = torch.flatten(features, 1)
            
        alpha = torch.exp(self.g_alpha(features))
        p = F.softmax(self.g_p(features), dim=1)
        tau = F.softplus(self.g_tau(features))

        return alpha, p, tau

###############################################
# Utility Layers
###############################################

class AvgPoolShortCut(nn.Module):
    def __init__(self, stride, out_c, in_c):
        super(AvgPoolShortCut, self).__init__()
        self.stride = stride
        self.out_c = out_c
        self.in_c = in_c

    def forward(self, x):
        if x.shape[2] % 2 != 0:
            x = F.avg_pool2d(x, 1, self.stride)
        else:
            x = F.avg_pool2d(x, self.stride, self.stride)
        pad = torch.zeros(x.shape[0], self.out_c - self.in_c, x.shape[2], x.shape[3], device=x.device)
        x = torch.cat((x, pad), dim=1)
        return x


class SpectralLinear(nn.Module):
    def __init__(self, input_dim, output_dim, k_lipschitz=1.0):
        super().__init__()
        self.k_lipschitz = k_lipschitz
        self.spectral_linear = spectral_norm(nn.Linear(input_dim, output_dim))

    def forward(self, x):
        return self.k_lipschitz * self.spectral_linear(x)


class SpectralConv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_dim, padding, k_lipschitz=1.0):
        super().__init__()
        self.k_lipschitz = k_lipschitz
        self.spectral_conv = spectral_norm(nn.Conv2d(input_dim, output_dim, kernel_dim, padding=padding))

    def forward(self, x):
        return self.k_lipschitz * self.spectral_conv(x)


###############################################
# Sequential Construction Utilities
###############################################

def linear_sequential(input_dims, hidden_dims, output_dim, k_lipschitz=None, p_drop=None):
    dims = [np.prod(input_dims)] + hidden_dims + [output_dim]
    num_layers = len(dims) - 1
    layers = []
    for i in range(num_layers):
        if k_lipschitz is not None:
            l = SpectralLinear(dims[i], dims[i + 1], k_lipschitz ** (1. / num_layers))
            layers.append(l)
        else:
            layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < num_layers - 1:
            layers.append(nn.ReLU())
            if p_drop is not None:
                layers.append(nn.Dropout(p=p_drop))
    return nn.Sequential(*layers)


def convolution_sequential(input_dims, hidden_dims, output_dim, kernel_dim, k_lipschitz=None, p_drop=None):
    channel_dim = input_dims[2]
    dims = [channel_dim] + hidden_dims
    num_layers = len(dims) - 1
    layers = []
    for i in range(num_layers):
        if k_lipschitz is not None:
            l = SpectralConv(dims[i], dims[i + 1], kernel_dim, (kernel_dim - 1) // 2, k_lipschitz ** (1. / num_layers))
            layers.append(l)
        else:
            layers.append(nn.Conv2d(dims[i], dims[i + 1], kernel_dim, padding=(kernel_dim - 1) // 2))
        layers.append(nn.ReLU())
        if p_drop is not None:
            layers.append(nn.Dropout(p=p_drop))
        layers.append(nn.MaxPool2d(2, padding=0))
    return nn.Sequential(*layers)


###############################################
# Core CNN + Linear Hybrid
###############################################

class ConvLinSeq(nn.Module):
    def __init__(self, input_dims, linear_hidden_dims, conv_hidden_dims, output_dim, kernel_dim, batch_size, k_lipschitz, p_drop):
        super().__init__()
        if k_lipschitz is not None:
            k_lipschitz = k_lipschitz ** (1. / 2.)
        self.convolutions = convolution_sequential(input_dims=input_dims,
                                                   hidden_dims=conv_hidden_dims,
                                                   output_dim=output_dim,
                                                   kernel_dim=kernel_dim,
                                                   k_lipschitz=k_lipschitz,
                                                   p_drop=p_drop)
        self.linear = linear_sequential(
            input_dims=[conv_hidden_dims[-1] * (input_dims[0] // 2 ** len(conv_hidden_dims)) * (input_dims[1] // 2 ** len(conv_hidden_dims))],
            hidden_dims=linear_hidden_dims,
            output_dim=output_dim,
            k_lipschitz=k_lipschitz,
            p_drop=p_drop)

    def forward(self, input):
        batch_size = input.size(0)
        input = self.convolutions(input)
        self.feature = input.clone().detach().reshape(batch_size, -1)
        input = self.linear(input.reshape(batch_size, -1))
        return input


def convolution_linear_sequential(input_dims, linear_hidden_dims, conv_hidden_dims, output_dim, kernel_dim, batch_size, k_lipschitz, p_drop=None):
    return ConvLinSeq(input_dims=input_dims,
                      linear_hidden_dims=linear_hidden_dims,
                      conv_hidden_dims=conv_hidden_dims,
                      output_dim=output_dim,
                      kernel_dim=kernel_dim, batch_size=batch_size,
                      k_lipschitz=k_lipschitz,
                      p_drop=p_drop)


###############################################
# VGG Architecture
###############################################

class VGG(nn.Module):
    def __init__(self, features, output_dim, p_drop, k_lipschitz=None):
        super(VGG, self).__init__()
        self.features = features
        if k_lipschitz is not None:
            self.classifier = nn.Sequential(
                nn.Dropout(p=p_drop),
                SpectralLinear(512, 256, k_lipschitz),
                nn.ReLU(True),
                nn.BatchNorm1d(256),
                nn.Dropout(p=p_drop),
                SpectralLinear(256, output_dim, k_lipschitz),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(p=p_drop),
                nn.Linear(512, 256),
                nn.ReLU(True),
                nn.BatchNorm1d(256),
                nn.Dropout(p=p_drop),
                nn.Linear(256, output_dim),
            )

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        self.feature = x.clone().detach()
        return self.classifier(x)


def make_layers(cfg, dropout_rate, batch_norm=False, k_lipschitz=None):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2), nn.Dropout(p=dropout_rate)]
        else:
            conv2d = SpectralConv(in_channels, v, 3, padding=1, k_lipschitz=k_lipschitz) if k_lipschitz else nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'D': [64, 64, 'M', 128, 128, 'M',
          256, 256, 256, 'M', 512, 512, 512, 'M',
          512, 512, 512, 'M']
}


def vgg16_bn(output_dim, p_drop, k_lipschitz=None):
    if k_lipschitz is not None:
        k_lipschitz = k_lipschitz ** (1. / 16.)
    return VGG(make_layers(cfg['D'], p_drop, batch_norm=True, k_lipschitz=k_lipschitz),
               output_dim=output_dim, p_drop=p_drop, k_lipschitz=k_lipschitz)


###############################################
# Main Architectures
###############################################

def conv_net(p_drop, spect_norm=False):
    input_dims = [28, 28, 1]
    linear_hidden_dims = [64, 64]
    conv_hidden_dims = [64, 64, 64]
    output_dim = 10
    kernel_dim = 5
    batch_size = 64
    k_lipschitz = 1 if spect_norm else None
    return convolution_linear_sequential(input_dims, linear_hidden_dims, conv_hidden_dims,
                                         output_dim, kernel_dim, batch_size, k_lipschitz, p_drop)


def vgg16(output_dim, p_drop, spect_norm):
    k_lipschitz = 1 if spect_norm else None
    return vgg16_bn(output_dim, p_drop=p_drop, k_lipschitz=k_lipschitz)


###############################################
# ResNet
###############################################

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, input_size, wrapped_conv, in_planes, planes, stride=1, mod=True, dropout_rate=0):
        super(BasicBlock, self).__init__()
        self.conv1 = wrapped_conv(input_size, in_planes, planes, kernel_size=3, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = wrapped_conv(math.ceil(input_size / stride), planes, planes, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.mod = mod
        self.activation = F.leaky_relu if self.mod else F.relu

        if stride != 1 or in_planes != self.expansion * planes:
            if mod:
                self.shortcut = nn.Sequential(AvgPoolShortCut(stride, self.expansion * planes, in_planes))
            else:
                self.shortcut = nn.Sequential(
                    wrapped_conv(input_size, in_planes, self.expansion * planes, kernel_size=1, stride=stride),
                    nn.BatchNorm2d(self.expansion * planes),
                )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.dropout2(out)
        out += self.shortcut(x)
        return self.activation(out)


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100, temp=1.0, spectral_normalization=True, mod=True, mnist=False, dropout_rate=0):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.mod = mod
        self.temp = temp

        def wrapped_conv(input_size, in_c, out_c, kernel_size, stride):
            conv = nn.Conv2d(in_c, out_c, kernel_size, stride, padding=1 if kernel_size == 3 else 0, bias=False)
            return torch.nn.utils.spectral_norm(conv) if spectral_normalization else conv

        self.wrapped_conv = wrapped_conv
        self.bn1 = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout(p=dropout_rate)

        self.conv1 = wrapped_conv(32, 3 if not mnist else 1, 64, kernel_size=3, stride=1)
        self.layer1 = self._make_layer(block, 32, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 16, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 8, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AvgPool2d(kernel_size=4)
        self.features = nn.Sequential(
            self.conv1, self.bn1, self.layer1, self.dropout,
            self.layer2, self.dropout, self.layer3, self.dropout,
            self.layer4, self.dropout, self.avgpool
        )
        self.classifier = nn.Sequential(nn.Dropout(p=dropout_rate), nn.Linear(512 * block.expansion, num_classes))

    def _make_layer(self, block, input_size, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(input_size, self.wrapped_conv, self.in_planes, planes, s, self.mod))
            self.in_planes = planes * block.expansion
            input_size = math.ceil(input_size / s)
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        self.feature = x.clone().detach()
        return self.classifier(x) / self.temp


def resnet18(spectral_normalization, mod=True, temp=1.0, mnist=False, dropout_rate=0):
    return ResNet(BasicBlock, [2, 2, 2, 2], spectral_normalization=spectral_normalization,
                  mod=mod, temp=temp, mnist=mnist, dropout_rate=dropout_rate)




