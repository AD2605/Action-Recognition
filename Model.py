import math
import time

import torch
import torch.nn as nn
from torch.nn.modules.utils import _triple
import torch.backends.cudnn as cudnn
from apex import amp


class block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, padding):
        super(block, self).__init__()
        kernel = _triple(kernel)
        stride = _triple(stride)
        padding = _triple(padding)

        spatial_kernel = [1, kernel[1], kernel[2]]
        spatial_stride = [1, stride[1], stride[2]]
        spatial_pad = [0, padding[1], padding[2]]

        temporal_kernel = [kernel[0], 1, 1]
        temporal_stride = [stride[0], 1, 1]
        temporal_pad = [padding[0], 0, 0]

        # compute intermediate channels as given in the paper
        interim_channels = int(math.floor((kernel[0] * kernel[1] * kernel[2] * in_channels * out_channels) / (
                    (kernel[1] * kernel[2] * in_channels) + (kernel[0] * out_channels))))

        self.spatialConv = nn.Conv3d(in_channels=in_channels,
                                     out_channels=interim_channels,
                                     kernel_size=spatial_kernel,
                                     stride=spatial_stride,
                                     padding=spatial_pad,
                                     bias=True)

        self.temporalConv = nn.Conv3d(in_channels=interim_channels,
                                      out_channels=out_channels,
                                      kernel_size=temporal_kernel,
                                      stride=temporal_stride,
                                      padding=temporal_pad,
                                      bias=True)

        self.batchNorm = nn.BatchNorm3d(interim_channels, affine=True)

    def forward(self, x):
        x = self.spatialConv(x)
        x = self.batchNorm(x)
        x = nn.functional.relu(x)
        x = self.temporalConv(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1):
        super(ResidualBlock, self).__init__()
        pad = kernel // 2
        self.Conv1 = block(in_channels=in_channels, out_channels=out_channels, kernel=kernel, stride=stride,
                           padding=pad)
        self.Conv2 = block(in_channels=out_channels, out_channels=out_channels, kernel=kernel, stride=stride,
                           padding=pad)
        self.batchNorm = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        res = self.Conv1(x)
        res = self.batchNorm(res)
        res1 = nn.functional.relu(res)
        res = self.Conv2(res)
        res = self.batchNorm(res)
        out = nn.functional.relu(res + res1)
        return out


class R2plus1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, depth):
        super(R2plus1D, self).__init__()
        self.layer1 = ResidualBlock(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel=kernel)
        self.layers = nn.ModuleList([])
        for i in range(0, depth + 1):
            self.layers = self.layers.append(ResidualBlock(in_channels=out_channels,
                                                           out_channels=out_channels,
                                                           kernel=kernel))

    def forward(self, x):
        x = self.layer1(x)
        for layer in self.layers:
            x = layer(x)
        return x


class spatioTemporalClassifier(nn.Module):
    def __init__(self, classes):
        super(spatioTemporalClassifier, self).__init__()
        self.classes = classes
        self.linear = nn.Linear(in_features=1024, out_features=classes)
        self.conv1 = block(in_channels=3, out_channels=64, kernel=[3, 5, 5], stride=[1, 2, 2], padding=[1, 3, 3])
        self.conv2 = R2plus1D(in_channels=64, out_channels=64, kernel=3, depth=4)
        self.conv3 = R2plus1D(in_channels=64, out_channels=128, kernel=3, depth=4)
        self.conv4 = R2plus1D(in_channels=128, out_channels=256, kernel=3, depth=4)
        self.conv5 = R2plus1D(in_channels=256, out_channels=512, kernel=3, depth=4)
        self.pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool(x)
        x = x.view(-1, 1024)
        x = self.linear(x)
        return x

    def train_model(self, model, dataloader, epochs):
        cudnn.benchmark = True
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        model.train()
        model.cuda()
        model, optimizer = amp.initialize(model, optimizer, opt_level='O3', keep_batchnorm_fp32=False)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.1)
        # criterion = torch.nn.CrossEntropyLoss().cuda()
        criterion = torch.nn.BCELoss().cuda()
        for i in range(0, epochs):
            train_accuracy = 0
            net_loss = 0
            for _, (data, label) in enumerate(dataloader):
                optimizer.zero_grad()
                data = data.half().cuda()
                label = label.cuda()
                out = model(data)
                loss = criterion(out, label)
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                    optimizer.step()
                if torch.argmax(out) == label:
                    train_accuracy += 1
                net_loss += loss.item()
            print('------------------------------------------')
            print('EPOCH ', i)
            print(train_accuracy / len(dataloader))
            print(net_loss / len(dataloader))
            scheduler.step()

    def evaluate(self, model, dataloader):
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False
        correct = 0
        model.cuda()
        for _, (x, y) in enumerate(dataloader):
            x = x.cuda()
            y = y.cuda()
            out = model(x)
            if torch.argmax(out) == y:
                correct += 1
        print(correct / len(dataloader))
