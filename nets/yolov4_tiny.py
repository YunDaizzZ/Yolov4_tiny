# coding: utf-8
from __future__ import division
import torch
import torch.nn as nn
from nets.CSPdarknet53_tiny import darknet53_tiny

# 卷积块 conv + batchnorm + leakyrelu
class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        return x

# 卷积 + 上采样
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()
        self.upsample = nn.Sequential(
            BasicConv(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest'))

    def forward(self, x):
        x = self.upsample(x)

        return x

# 获得yolov4的输出
def yolo_head(filters_list, in_filters):
    m = nn.Sequential(
        BasicConv(in_filters, filters_list[0], 3),
        nn.Conv2d(filters_list[0], filters_list[1], 1))

    return m

# yolo_body
class YoloBody(nn.Module):
    def __init__(self, num_anchors, num_classes):
        super(YoloBody, self).__init__()
        # backbone
        self.backbone = darknet53_tiny(None)

        self.conv_for_P5 = BasicConv(512, 256, 1)
        self.yolo_headP5 = yolo_head([512, num_anchors * (5 + num_classes)], 256)

        self.upsample = Upsample(256, 128)
        self.yolo_headP4 = yolo_head([256, num_anchors * (5 + num_classes)], 384)

    def forward(self, x):
        feat1, feat2 = self.backbone(x)

        p5 = self.conv_for_P5(feat2)
        out0 = self.yolo_headP5(p5)

        p5_up = self.upsample(p5)
        p4 = torch.cat([p5_up, feat1], axis=1)

        out1 = self.yolo_headP4(p4)

        return out0, out1
