from model import common
import torch
import torch.nn as nn


def make_model(args, parent=False):
    if args.dilation:
        from model import dilated
        return UNET(args, dilated.dilated_conv)
    else:
        return UNET(args)

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.2,inplace=True),
            nn.BatchNorm2d(out_ch),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.2,inplace=True),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, input):
        return self.conv(input)

class UP(nn.Module):
    def __init__(
        self):
        super(UP, self).__init__()

    def forward(self, x):
        x1 = nn.functional.interpolate(x,scale_factor=2, mode='bilinear', align_corners=True)
        return x1

class RB(nn.Module):
    def __init__(
        self, in_ch, out_ch,act=nn.ReLU(True)):
        super(RB, self).__init__()
        m = []
        m.append(nn.Conv2d(in_ch, out_ch, 3, padding=1))
        m.append(act)
        m.append(nn.Conv2d(out_ch, out_ch, 3, padding=1))
        self.body = nn.Sequential(*m)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class UNET(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(UNET, self).__init__()

        m_1 = [DoubleConv(args.n_colors+1, 32)]
        m_1234 = [nn.MaxPool2d(2)]
        m_2 = [DoubleConv(32, 64)]
        m_3 = [DoubleConv(64, 128)]
        m_4 = [DoubleConv(128, 256)]

        m_5 = [
            DoubleConv(256, 512),
            UP(),
            nn.Conv2d(512, 256, 3, padding=1)
        ]
        m_6 = [
            nn.Conv2d(512, 256,1),
            nn.BatchNorm2d(256),
            RB(256,256),
            RB(256,256),
            UP(),
            nn.Conv2d(256, 128, 3, padding=1)
        ]
        m_7 = [
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            RB(128, 128),
            RB(128, 128),
            UP(),
            nn.Conv2d(128, 64, 3, padding=1)
        ]
        m_8 = [
            nn.Conv2d(128, 64, 1),
            nn.BatchNorm2d(64),
            RB(64, 64),
            RB(64, 64),
            UP(),
            nn.Conv2d(64, 32, 3, padding=1)
        ]
        m_9 = [
            nn.Conv2d(64, 32, 1),
            RB(32, 32),
            RB(32, 32)]
        m_10 = [nn.Conv2d(32, 3, 1)]

        self.m1 = nn.Sequential(*m_1)
        self.m1234 = nn.Sequential(*m_1234)
        self.m2 = nn.Sequential(*m_2)
        self.m3 = nn.Sequential(*m_3)
        self.m4 = nn.Sequential(*m_4)
        self.m5 = nn.Sequential(*m_5)
        self.m6 = nn.Sequential(*m_6)
        self.m7 = nn.Sequential(*m_7)
        self.m8 = nn.Sequential(*m_8)
        self.m9 = nn.Sequential(*m_9)
        self.m10 = nn.Sequential(*m_10)

    def forward(self, x):
        x1 = self.m1(x)
        xp1 = self.m1234(x1)
        x2 = self.m2(xp1)
        xp2 = self.m1234(x2)
        x3 = self.m3(xp2)
        xp3 = self.m1234(x3)
        x4 = self.m4(xp3)
        xp4 = self.m1234(x4)
        x5 = torch.cat([x4, self.m5(xp4)], 1)
        x6 = torch.cat([x3, self.m6(x5)], 1)
        x7 = torch.cat([x2, self.m7(x6)], 1)
        x8 = torch.cat([x1, self.m8(x7)], 1)
        x9 = self.m9(x8)
        x = self.m10(x9)
        x = torch.sigmoid(x)
        return x

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

