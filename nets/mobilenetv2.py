import os
import math
import torch
import torch.nn as nn


###### 构建mobilenetv2骨架网络 ######


# 3 * 3卷积
def conv_bn(inplane, outplane, stride):
    return nn.Sequential(
        nn.Conv2d(inplane, outplane, 3, stride, 1, bias=False),
        nn.BatchNorm2d(outplane),
        nn.ReLU6(inplace=True),
    )


# 1 * 1卷积
def conv_1_bn(inplane, outplane):
    return nn.Sequential(
        nn.Conv2d(inplane, outplane, 1, 1, 0, bias=False),
        nn.BatchNorm2d(outplane),
        nn.ReLU6(inplace=True),
    )


# 残差结构
class InvertedResidual(nn.Module):
    def __init__(self, inplane, outplane, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride

        hidden_dim = round(inplane * expand_ratio)  # 中间隐藏层的数目
        self.use_res_connect = self.stride == 1 and inplane == outplane  # 是否使用shortcut连接

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups= hidden_dim, bias=False),  # DW卷积
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),

                # 1 * 1卷积上升通道数
                nn.Conv2d(hidden_dim, outplane, 1, 1, 0, bias=False),  # PW卷积
                nn.BatchNorm2d(outplane),
            )
        else:
            self.conv = nn.Sequential(
                # 1 * 1卷积上升通道数
                nn.Conv2d(inplane, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),

                # 3 * 3卷积
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),  # DW卷积
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),

                # 1 * 1卷积降维
                nn.Conv2d(hidden_dim, outplane, 1, 1, 0, bias=False), # PW卷积
                nn.BatchNorm2d(outplane),
            )
    
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)  # short_cut连接
        else:
            return self.conv(x)


# 定义主干网络
class MobileNetV2(nn.Module):
    def __init__(self):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        # 残差block
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],  # 256, 256, 32 -> 256, 256, 16  
            [6, 24, 2, 2],  # 256, 256, 16 -> 128, 128, 24  2
            [6, 32, 3, 2],  # 128, 128, 24 -> 64, 64, 32  4
            [6, 64, 4, 2],  # 64, 64, 32 -> 32, 32, 64  7
            [6, 96, 3, 1],  # 32, 32, 64 -> 32, 32, 96
            [6, 160, 3, 2], # 32, 32, 96 -> 16, 16, 160  14
            [6, 320, 1, 1], # 16, 16, 160 -> 16, 16, 320
        ]
        self.last_channel = last_channel
        self.features = [conv_bn(3, input_channel, 2)]  # 512, 512, 3 -> 256, 256, 32

        # 构建残差block
        for t, c, n, s in interverted_residual_setting:
            output_channel = c
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        
        # 16, 16, 320 -> 16, 16, 1280
        self.features.append(conv_1_bn(input_channel, self.last_channel))
        self.features = nn.Sequential(*self.features)

    def forward(self, x):
        x = self.features(x)
        return x


# 定义mobilenetv2网络
def mobilenetv2():
    model = MobileNetV2()
    return model