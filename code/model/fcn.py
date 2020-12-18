from model import common
import torch
import torch.nn as nn


def make_model(args, parent=False):
    if args.dilation:
        from model import dilated
        return FCN(args, dilated.dilated_conv)
    else:
        return FCN(args)

class FCN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(FCN, self).__init__()
        # conv1
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2)  # 1/2

        # conv2
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2)  # 1/4

        # conv3
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2)  # 1/8

        # conv4
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2)  # 1/16

        self.upscore16 = nn.ConvTranspose2d(
            64, 64, 16, stride=16, bias=False)

        # conv5
        self.conv5 = nn.Conv2d(64, 64, 3,padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        # conv6
        self.conv6 = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        h = x
        h = self.relu1(self.conv1(h))
        h = self.pool1(h)

        h = self.relu2(self.conv2(h))
        h = self.pool2(h)

        h = self.relu3(self.conv3(h))
        h = self.pool3(h)

        h = self.relu4(self.conv4(h))
        h = self.pool4(h)

        h = self.upscore16(h)

        h = self.relu5(self.conv5(h))
        h = self.conv6(h)
        return h

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

