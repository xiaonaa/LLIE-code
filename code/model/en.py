from model import common
import torch
import torch.nn as nn


def make_model(args,ops, parent=False):
    if args.dilation:
        from model import dilated
        return EN(args,ops, dilated.dilated_conv)
    else:
        return EN(args,ops)

class EN(nn.Module):
    def __init__(self, args,ops, conv=common.default_conv):
        super(EN, self).__init__()
        if ops == 'U':
            c_in = 4
            c_out = 1
        elif ops == 'N':
            c_in = 4
            c_out = 3
        # conv1
        self.conv1 = nn.Conv2d(c_in, 64, 3, padding=1)
        self.lrelu1 = nn.LeakyReLU(negative_slope=0.2,inplace=True)

        # conv2
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.lrelu2 = nn.LeakyReLU(negative_slope=0.2,inplace=True)

        # conv3
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.lrelu3 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # conv4
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.lrelu4 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # conv5
        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
        self.lrelu5 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # conv6
        self.conv6 = nn.Conv2d(64, 64, 3, padding=1)
        self.lrelu6 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # conv7
        self.conv7 = nn.Conv2d(64, c_out, 3, padding=1)


    def forward(self, x):
        h = x
        h = self.lrelu1(self.conv1(h))
        h = self.lrelu2(self.conv2(h))
        h = self.lrelu3(self.conv3(h))
        h = self.lrelu4(self.conv4(h))
        h = self.lrelu5(self.conv5(h))
        h = self.lrelu6(self.conv6(h))
        h = self.conv7(h)
        h = torch.sigmoid(h)
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

