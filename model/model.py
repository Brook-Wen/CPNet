import math
import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

from .resnet import resnet50


class ConvertLayer(nn.Module):
    def __init__(self, list_in, list_out):
        super(ConvertLayer, self).__init__()
        self.convert = nn.ModuleList()
        if isinstance(list_out, list):
            for i in range(len(list_in)):
                self.convert.append(nn.Sequential(nn.Conv2d(list_in[i], list_out[i], kernel_size=1, stride=1, padding=0, bias=False),
                                                nn.ReLU(inplace=True)))
        else:
            for i in range(len(list_in)):
                self.convert.append(nn.Sequential(nn.Conv2d(list_in[i], list_out, kernel_size=1, stride=1, padding=0, bias=False),
                                                nn.ReLU(inplace=True)))

    def forward(self, x):
        y = []
        [y.append(self.convert[i](x[i])) for i in range(len(x))]
        return y


# Patch Partition Module
class PatchPartition(nn.Module):
    def __init__(self):
        super(PatchPartition, self).__init__()

    def forward(self, x):
        x = list(torch.chunk(x, 4, dim=3))
        y = []
        [y.append(torch.cat([x[i], x[i+1]], dim=3)) if i < 3 else y.append(torch.cat([x[i], x[0]], dim=3)) for i in range(4)]
        y = torch.cat(y, dim=0)

        return y


# Reverse Patch Partition Module
class ReversePatchPartition(nn.Module):
    def __init__(self):
        super(ReversePatchPartition, self).__init__()

    def forward(self, x):
        x = list(torch.chunk(x, 4, dim=0))
        x_ = torch.chunk(x[0], 2, dim=3)
        x_0_0, x_1_1 = x_[0], x_[1]
        x_ = torch.chunk(x[1], 2, dim=3)
        x_0_1, x_1_2 = x_[0], x_[1]
        x_ = torch.chunk(x[2], 2, dim=3)
        x_0_2, x_1_3 = x_[0], x_[1]
        x_ = torch.chunk(x[3], 2, dim=3)
        x_0_3, x_1_0 = x_[0], x_[1]

        x_0 = torch.cat([x_0_0, x_0_1, x_0_2, x_0_3], dim=3)
        x_1 = torch.cat([x_1_0, x_1_1, x_1_2, x_1_3], dim=3)
        x = x_0 + x_1

        return x


# Consistency Context-aware Module
class ConsistencyContextAware(nn.Module):
    def __init__(self, planes, num_levels=4):
        super(ConsistencyContextAware, self).__init__()
        self.planes = planes
        self.num_levels = num_levels
        self.hw = (7, 14)
        self.scale = self.planes ** -0.5

        self.conv_q = nn.Conv2d(self.planes, self.planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_kv = nn.Conv2d(self.planes, self.planes*2, kernel_size=1, stride=1, padding=0, bias=False)
        self.proj_ga = nn.Conv2d(self.planes, self.planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_sr = nn.Conv2d(self.planes, self.planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.pool = nn.AdaptiveAvgPool2d(self.hw)
        self.norm = nn.LayerNorm(self.planes)
        self.gelu = nn.GELU()

        self.conv_sa = nn.Conv2d(self.planes, 1, kernel_size=(3, 5), stride=1, padding=(1, 2), bias=False)
        self.proj_la = nn.Conv2d(self.planes, self.planes, kernel_size=1, stride=1, padding=0, bias=False)
        if self.num_levels > 1:
            self.conv_sam = nn.Conv2d(self.num_levels-1, 1, kernel_size=(3, 5), stride=1, padding=(1, 2), bias=False)
            self.conv_cat = nn.Conv2d(self.planes*(self.num_levels-1), self.planes, kernel_size=1, stride=1, padding=0, bias=False)
            self.conv_sum = nn.Sequential(nn.Conv2d(self.planes, self.planes, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=False),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(self.planes, self.planes, kernel_size=(1, 5), stride=1, padding=(0, 2), bias=False))

        self.conv = nn.Sequential(nn.Conv2d(self.planes, self.planes, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=False),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(self.planes, self.planes, kernel_size=(1, 5), stride=1, padding=(0, 2), bias=False))
        self.relu = nn.ReLU(inplace=True)

    def local_attention(self, x, x_c=None):
        sam = self.conv_sa(x).sigmoid()
        if self.num_levels > 1:
            x = x + x_c
            x = self.relu(self.conv_sum(x))
        x = x + x * sam
        x = self.relu(self.proj_la(x))
        return x

    def global_attention(self, x):
        b, c, h, w = x.size()
        x_q = self.conv_q(x).view(b, c, -1).permute(0, 2, 1) # b, h*w, c
        x_kv = self.conv_sr(self.pool(x)).view(b, c, -1).permute(0, 2, 1) # b, h'*w', c
        x_kv = self.gelu(self.norm(x_kv)).view(b, c, self.hw[0], self.hw[1]) # b, c, h', w'
        x_kv = self.conv_kv(x_kv).view(b, 2, c, -1).permute(1, 0, 2, 3) # 2, b, c, h'*w'
        x_k, x_v = x_kv[0], x_kv[1] # b, c, h'*w'

        att = (x_q @ x_k) * self.scale # b, h*w, h'*w'
        att = att.softmax(dim=-1)
        x = (x_v @ att.permute(0, 2, 1)).view(b, c, h, w) # b, c, h, w
        x = self.relu(self.proj_ga(x))
        return x

    def forward(self, x, mask=None):
        sal_size = x[0].size()
        if self.num_levels > 1:
            x_c = [F.interpolate(x[i], sal_size[2:], mode='bilinear', align_corners=True) for i in range(1, self.num_levels)]
            x_c = torch.cat(x_c, dim=1)
            x_c = self.relu(self.conv_cat(x_c))

            x_m = [F.interpolate(mask[i], sal_size[2:], mode='bilinear', align_corners=True) for i in range(self.num_levels-1)]
            x_m = torch.cat(x_m, dim=1)
            sam = self.conv_sam(x_m).sigmoid()

            x_la = self.local_attention(x[0], x_c)
            x_ga = self.global_attention(x_la)
            x = (x_la + x_ga) * sam
        else:
            x_la = self.local_attention(x[0])
            x_ga = self.global_attention(x_la)
            x = x_la + x_ga
        x = self.relu(self.conv(x))

        return x


# Bidirectional Scale-aware Module
class BidirectionalScaleAware(nn.Module):
    def __init__(self, planes, dila_ratio=[1, 2, 3, 5]):
        super(BidirectionalScaleAware, self).__init__()
        self.planes = planes
        self.num_groups = len(dila_ratio)
        self.mid_planes = self.planes // self.num_groups
        self.dila_ratio1 = dila_ratio
        self.dila_ratio2 = dila_ratio[::-1]

        self.conv_dila1 = nn.ModuleList()
        self.conv_dila2 = nn.ModuleList()
        for i in range(self.num_groups):
            self.conv_dila1.append(nn.Sequential(nn.Conv2d(self.mid_planes*(i+1), self.mid_planes, kernel_size=1, stride=1, padding=0, bias=False),
                                                nn.ReLU(inplace=True),
                                                nn.Conv2d(self.mid_planes, self.mid_planes, kernel_size=(3, 1), stride=1, padding=(self.dila_ratio1[i], 0), dilation=(self.dila_ratio1[i], 1), bias=False),
                                                nn.ReLU(inplace=True),
                                                nn.Conv2d(self.mid_planes, self.mid_planes, kernel_size=(1, 5), stride=1, padding=(0, self.dila_ratio1[i]*2), dilation=(1, self.dila_ratio1[i]), bias=False)))
            self.conv_dila2.append(nn.Sequential(nn.Conv2d(self.mid_planes*(i+1), self.mid_planes, kernel_size=1, stride=1, padding=0, bias=False),
                                                nn.ReLU(inplace=True),
                                                nn.Conv2d(self.mid_planes, self.mid_planes, kernel_size=(3, 1), stride=1, padding=(self.dila_ratio2[i], 0), dilation=(self.dila_ratio2[i], 1), bias=False),
                                                nn.ReLU(inplace=True),
                                                nn.Conv2d(self.mid_planes, self.mid_planes, kernel_size=(1, 5), stride=1, padding=(0, self.dila_ratio2[i]*2), dilation=(1, self.dila_ratio2[i]), bias=False)))

        self.conv_1 = nn.Conv2d(self.planes, self.planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(self.planes, self.planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_3 = nn.Sequential(nn.Conv2d(self.planes, self.planes, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=False),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(self.planes, self.planes, kernel_size=(1, 5), stride=1, padding=(0, 2), bias=False))
        self.alpha = nn.Parameter(torch.ones(1), requires_grad=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = list(torch.chunk(x, self.num_groups, dim=1))
        for i in range(self.num_groups):
            if i == 0:
                y = x[i]
                y = self.conv_dila1[i](y)
                y_1 = y

                y = x[i]
                y = self.conv_dila2[i](y)
                y_2 = y
            else:
                y = torch.cat([x[i], y_1], dim=1)
                y = self.conv_dila1[i](self.relu(y))
                y_1 = torch.cat([y_1, y], dim=1)

                y = torch.cat([x[i], y_2], dim=1)
                y = self.conv_dila2[i](self.relu(y))
                y_2 = torch.cat([y_2, y], dim=1)

        y_1 = self.conv_1(self.relu(y_1))
        y_2 = self.conv_2(self.relu(y_2))
        y = y_1 * self.alpha + y_2 * (1 - self.alpha)
        y = self.relu(self.conv_3(self.relu(y)))

        return y


class Decoder(nn.Module):
    def __init__(self, planes, num_levels=4):
        super(Decoder, self).__init__()
        self.planes = planes
        self.num_levels = num_levels

        self.bsa = nn.ModuleList()
        self.cca = nn.ModuleList()
        self.loss_s = nn.ModuleList()
        self.loss_e = nn.ModuleList()
        for i in range(self.num_levels):
            self.bsa.append(BidirectionalScaleAware(self.planes))
            self.cca.append(ConsistencyContextAware(self.planes, num_levels=i+1))
            self.loss_s.append(nn.Conv2d(self.planes, 1, kernel_size=1, stride=1, padding=0, bias=False))
            self.loss_e.append(nn.Conv2d(self.planes, 1, kernel_size=1, stride=1, padding=0, bias=False))

        self.maxpool_dilate = nn.MaxPool2d(kernel_size=7, stride=1, padding=3, ceil_mode=True)
        self.maxpool_erode = nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True)
        self.deconv = nn.Sequential(nn.Conv2d(self.planes, self.planes//2, kernel_size=1, stride=1, padding=0, bias=False),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(self.planes//2, self.planes//2, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=False),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(self.planes//2, self.planes//2, kernel_size=(1, 5), stride=1, padding=(0, 2), bias=False),
                                    nn.ReLU(inplace=True))
        self.loss = nn.Conv2d(self.planes//2, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, sal_size):
        loss_s, loss_e = [], []
        mask = None

        x = [self.bsa[i](x[i]) for i in range(self.num_levels)]
        for i in range(self.num_levels):    
            x[i] = self.cca[i](x[:i+1][::-1], mask)

            x_s = self.loss_s[i](x[i])
            loss_s.append(F.interpolate(x_s, sal_size[2:], mode='bilinear', align_corners=True))
            # Edge Sharpening Module
            x_h = (self.sigmoid(x_s) * (1 - self.sigmoid(x_s))) * 4
            x_e = self.maxpool_dilate(x[i]) - torch.neg(self.maxpool_erode(torch.neg(x[i])))
            loss_e.append(F.interpolate(self.loss_e[i](x_e * x_h), sal_size[2:], mode='bilinear', align_corners=True))
            mask = loss_s[::-1]

        x = F.interpolate(x[-1], scale_factor=2, mode='bilinear', align_corners=True)
        x = self.deconv(x)
        x = F.interpolate(x, sal_size[2:], mode='bilinear', align_corners=True)
        x = self.loss(x)

        return x, loss_s, loss_e


# Consistency Perception Network
class Model_bone(nn.Module):
    def __init__(self, base):
        super(Model_bone, self).__init__()
        self.c_list = 64
        self.convert = ConvertLayer([256, 512, 1024, 2048], self.c_list)

        self.base = base
        self.pp = PatchPartition()
        self.rpp = ReversePatchPartition()
        self.decoder = Decoder(self.c_list)

    def forward(self, x):
        sal_size = x.size()

        x = self.pp(x)
        x = self.base(x)
        x = self.convert(x)

        x = [self.rpp(x[i]) for i in range(len(x))]

        y, loss_s, loss_e = self.decoder(x[::-1], sal_size)

        return y, loss_s, loss_e


# build the whole network
def build_model():
    return Model_bone(resnet50())


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.zero_()
    elif isinstance(m, nn.LayerNorm):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.GroupNorm):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Embedding):
        m.weight.data.normal_(0.0, 0.01)
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0.0, 0.01)
        if m.bias is not None:
            m.bias.data.zero_()

