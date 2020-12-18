from model import common

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable

class VGG2(nn.Module):
    def __init__(self,rgb_range=1):
        super(VGG2, self).__init__()
        vgg_features = models.vgg19(pretrained=True).features
        modules = [m for m in vgg_features]
        self.vgg = nn.Sequential(*modules[:8])
        self.vgg1 = nn.Sequential(*modules[:35])
        rgb_range = 1
        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std = (0.229 , 0.224 , 0.225 )
        self.sub_mean = common.MeanShift(rgb_range, vgg_mean, vgg_std)
        self.vgg.requires_grad = False
        self.vgg1.requires_grad = False

    def forward(self, sr, hr):
        def _forward(x):
            x = self.sub_mean(x)
            x = self.vgg(x)
            return x
        def _forward1(x):
            x = self.sub_mean(x)
            x = self.vgg1(x)
            return x
        vgg_sr = _forward(sr)
        vgg_sr1 = _forward1(sr)
        with torch.no_grad():
            vgg_hr = _forward(hr.detach())
            vgg_hr1 = _forward1(hr.detach())

        loss = F.mse_loss(vgg_sr, vgg_hr)
        loss1 = F.mse_loss(vgg_sr1, vgg_hr1)
        loss = loss + loss1

        return loss
