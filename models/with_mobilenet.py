import numpy as np
from modules.conv import conv, conv_dw, conv_dw_no_bn

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import cv2


class Conv_bn_relu(nn.Module):
    def __init__(self, inp, oup, kernel_size=3, stride=1, pad=1, use_relu=True):
        super(Conv_bn_relu, self).__init__()
        self.use_relu = use_relu
        if self.use_relu:
            self.convs = nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size, stride, pad, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )
        else:
            self.convs = nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size, stride, pad, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        out = self.convs(x)
        return out


class StemBlock(nn.Module):
    def __init__(self, inp=3, num_init_features=32):
        super(StemBlock, self).__init__()

        self.stem_1 = Conv_bn_relu(inp, num_init_features, 3, 2, 1)

        self.stem_2a = Conv_bn_relu(num_init_features, int(num_init_features / 2), 1, 1, 0)

        self.stem_2b = Conv_bn_relu(int(num_init_features / 2), num_init_features, 3, 2, 1)

        self.stem_2p = nn.MaxPool2d(kernel_size=2, stride=2)

        self.stem_3 = Conv_bn_relu(num_init_features * 2, num_init_features, 1, 1, 0)

    def forward(self, x):
        stem_1_out = self.stem_1(x)

        stem_2a_out = self.stem_2a(stem_1_out)
        stem_2b_out = self.stem_2b(stem_2a_out)

        stem_2p_out = self.stem_2p(stem_1_out)

        out = self.stem_3(torch.cat((stem_2b_out, stem_2p_out), 1))

        return out


class DenseBlock(nn.Module):
    def __init__(self, inp, inter_channel, growth_rate):
        super(DenseBlock, self).__init__()

        self.cb1_a = Conv_bn_relu(inp, inter_channel, 1, 1, 0)
        self.cb1_b = Conv_bn_relu(inter_channel, growth_rate, 3, 1, 1)

        self.cb2_a = Conv_bn_relu(inp, inter_channel, 1, 1, 0)
        self.cb2_b = Conv_bn_relu(inter_channel, growth_rate, 3, 1, 1)
        self.cb2_c = Conv_bn_relu(growth_rate, growth_rate, 3, 1, 1)

    def forward(self, x):
        cb1_a_out = self.cb1_a(x)
        cb1_b_out = self.cb1_b(cb1_a_out)

        cb2_a_out = self.cb2_a(x)
        cb2_b_out = self.cb2_b(cb2_a_out)
        cb2_c_out = self.cb2_c(cb2_b_out)

        out = torch.cat((x, cb1_b_out, cb2_c_out), 1)

        return out


class TransitionBlock(nn.Module):
    def __init__(self, inp, oup, with_pooling=True):
        super(TransitionBlock, self).__init__()
        if with_pooling:
            self.tb = nn.Sequential(Conv_bn_relu(inp, oup, 1, 1, 0),
                                    nn.AvgPool2d(kernel_size=2, stride=2))
        else:
            self.tb = Conv_bn_relu(inp, oup, 1, 1, 0)

    def forward(self, x):
        out = self.tb(x)
        return out


class PeleeNet(nn.Module):
    def __init__(self, num_classes=1000, num_init_features=32, growthRate=32, nDenseBlocks=[3, 4, 8],
                 bottleneck_width=[1, 2, 4]):
        super(PeleeNet, self).__init__()

        self.stage = nn.Sequential()
        self.num_classes = num_classes
        self.num_init_features = num_init_features

        inter_channel = list()
        total_filter = list()
        dense_inp = list()

        self.half_growth_rate = int(growthRate / 2)

        # building stemblock
        self.stage.add_module('stage_0', StemBlock(3, num_init_features))

        #
        for i, b_w in enumerate(bottleneck_width):

            inter_channel.append(int(self.half_growth_rate * b_w / 4) * 4)

            if i == 0:
                total_filter.append(num_init_features + growthRate * nDenseBlocks[i])
                dense_inp.append(self.num_init_features)
            else:
                total_filter.append(total_filter[i - 1] + growthRate * nDenseBlocks[i])
                dense_inp.append(total_filter[i - 1])

            if i == len(nDenseBlocks) - 1:
                with_pooling = False
            else:
                with_pooling = True

            # building middle stageblock
            self.stage.add_module('stage_{}'.format(i + 1), self._make_dense_transition(dense_inp[i], total_filter[i],
                                                                                        inter_channel[i],
                                                                                        nDenseBlocks[i],
                                                                                        with_pooling=with_pooling)

                                  )

        self._initialize_weights()

    def _make_dense_transition(self, dense_inp, total_filter, inter_channel, ndenseblocks, with_pooling=True):
        layers = []

        for i in range(ndenseblocks):
            layers.append(DenseBlock(dense_inp, inter_channel, self.half_growth_rate))
            dense_inp += self.half_growth_rate * 2

        # Transition Layer without Compression
        if ndenseblocks == (3 or 4):
            layers.append(TransitionBlock(dense_inp, total_filter, with_pooling))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.stage(x)

        # global average pooling layer
        # x = F.avg_pool2d(x,kernel_size=7)
        # x = x.view(x.size(0), -1)
        out = x
        # x = self.classifier(x)

        out2 = F.log_softmax(x, dim=1)

        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


# if __name__ == '__main__':
#    p = PeleeNet(num_classes=1000)
#    input = torch.autograd.Variable(torch.ones(1, 3, 256, 344))
#    output = p(input)

#    print(output.size())

# torch.save(p.state_dict(), 'peleenet.pth.tar')


class Cpm(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.align = conv(in_channels, out_channels, kernel_size=1, padding=0, bn=False)
        self.trunk = nn.Sequential(
            conv_dw_no_bn(out_channels, out_channels),
            conv_dw_no_bn(out_channels, out_channels),
            conv_dw_no_bn(out_channels, out_channels)
        )
        self.conv = conv(out_channels, out_channels, bn=False)

    def forward(self, x):
        x = self.align(x)
        x = self.conv(x + self.trunk(x))
        return x


class RefinementStageBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.initial = conv(in_channels, out_channels, kernel_size=1, padding=0, bn=False)
        self.trunk = nn.Sequential(
            conv(out_channels, out_channels),
            conv(out_channels, out_channels, dilation=2, padding=2)
        )

    def forward(self, x):
        initial_features = self.initial(x)
        trunk_features = self.trunk(initial_features)
        return initial_features + trunk_features

class PAFStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_pafs):
        super().__init__()
        self.trunk = nn.Sequential(
            RefinementStageBlock(in_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels)
        )

        self.pafs = nn.Sequential(
            conv(out_channels, 512, kernel_size=1, padding=0, bn=False),
            conv(512, num_pafs, kernel_size=1, padding=0, bn=False, relu=False)
        )

    def forward(self, x):
        trunk_features = self.trunk(x)
        pafs = self.pafs(trunk_features)
        return [pafs]

class RefinementStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_heatmaps):
        super().__init__()
        self.trunk = nn.Sequential(
            RefinementStageBlock(in_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels)
        )
        self.heatmaps = nn.Sequential(
            conv(out_channels, out_channels, kernel_size=1, padding=0, bn=False),
            conv(out_channels, num_heatmaps, kernel_size=1, padding=0, bn=False, relu=False)
        )

    def forward(self, x):
        trunk_features = self.trunk(x)
        heatmaps = self.heatmaps(trunk_features)
        return [heatmaps]


class PoseEstimationWithMobileNet(nn.Module):
    def __init__(self, num_refinement_stages=1, num_channels=128, num_heatmaps=19, num_pafs=38):
        super().__init__()
        self.num_re = num_refinement_stages
        self.model = PeleeNet(num_classes=1000)
        """self.model = nn.Sequential(
            conv(     3,  32, stride=2, bias=False),
            conv_dw( 32,  64),
            conv_dw( 64, 128, stride=2),
            conv_dw(128, 128),
            conv_dw(128, 256, stride=2),
            conv_dw(256, 256),
            conv_dw(256, 512),  # conv4_2
            conv_dw(512, 512, dilation=2, padding=2),
            conv_dw(512, 512),
            conv_dw(512, 512),
            conv_dw(512, 512),
            conv_dw(512, 512)   # conv5_5
        )"""
        self.cpm = Cpm(512, num_channels)
        self.num_re = num_refinement_stages
        self.initial_stage = PAFStage(num_channels, num_channels, num_pafs)
        self.paf_stage = nn.ModuleList()
        for id in range(num_refinement_stages):
            if id==0:
                b=num_pafs
            else:
                b=num_pafs
            self.paf_stage.append(PAFStage(num_channels + b, num_channels, num_pafs))

        self.refinement_stages = nn.ModuleList()
        for idx in range(num_refinement_stages):
            if idx ==0:
                a=num_pafs
            else:
                a=num_heatmaps
            self.refinement_stages.append(RefinementStage(num_channels + a, num_channels,
                                                          num_heatmaps))

    def forward(self, x):
        backbone_features = self.model(x)
        #print(backbone_features.shape)
        backbone_features = self.cpm(backbone_features)

        aa=0
        for paf_stage in self.paf_stage:
            if aa==0:
                stages_output = self.initial_stage(backbone_features)
            else:
                stages_output.extend(
                    paf_stage(torch.cat([backbone_features,stages_output[-1]], dim=1)))
            aa+=1

        for refinement_stage in self.refinement_stages:
            stages_output.extend(
                refinement_stage(torch.cat([backbone_features, stages_output[-1]], dim=1)))
        return stages_output
