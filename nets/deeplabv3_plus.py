import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from nets.mobilenetv2 import mobilenetv2


###### 构建deeplabv3_plus网络 ######


# 修改MObileNetV2网络
class MobileNetV2(nn.Module):
    def __init__(self):
        super(MobileNetV2, self).__init__()

        model = mobilenetv2()
        self.features = model.features[:-1]  # 取到最后一层特征提取层之前

        self.total_idx = len(self.features)  # 总的特征层数
        self.down_idx = [2, 4, 7, 14]  # 下采样层的index
        
        # 应用空洞卷积
        for i in range(self.down_idx[-1], self.total_idx):
            self.features[i].apply(
                partial(self._nostride_dilate, dilate=2)
            )
        
    # 空洞卷积   
    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__  # 获取此时的类名
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)  # 将步长为2的卷积改为步长为1,不降分辨率
                if m.kernel_size == (3, 3):  # 若卷积核为3 * 3
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)  # 采用填充为 dilate//2的空洞卷积
                
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)  # 采用填充为 dilate的空洞卷积，因为只有一次下采样，不用使用HDC
                    m.padding = (dilate, dilate)
    
    # 正向传播
    def forward(self, x):
        low_level_features = self.features[:4](x)  # 低等级特征, 用于构建encoder,decoder结构
        x = self.features[4:](low_level_features)
        return low_level_features, x
    

# 构建ASPP模块
# 利用不同膨胀率的膨胀卷积提取不同尺度的特征
class ASPP(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1):
        super(ASPP, self).__init__()

        # 1 * 1卷积
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),  # 应该设置为False
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
        )

        # 3 * 3卷积, 膨胀率为6
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
        )

        # 3 * 3卷积, 膨胀率为12
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
        )

        # 3 * 3卷积, 膨胀率为18
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=18 * rate, dilation= 18 * rate, bias=True),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
        )

        # 1 * 1卷积, 作用于全局平均池化后的特征图
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out)
        self.branch5_relu = nn.ReLU(inplace=True)

        # 1 * 1卷积, 作用于拼接后的向量
        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        [b, c, row, col] = x.size()  # b: batch_size, c: 通道数, row: 行, col: 列

        conv_1 = self.branch1(x)  # 1 * 1卷积
        conv_3_1 = self.branch2(x)  # 膨胀率为6的 3 * 3卷积
        conv_3_2 = self.branch3(x)  # 膨胀率为12的 3 * 3卷积
        conv_3_3 = self.branch4(x)  # 膨胀率为18的 3 * 3卷积

        global_feature = torch.mean(x, 2, True)  # 沿着行取均值
        global_feature = torch.mean(global_feature, 3, True)  # 沿着列取均值  处理后变为 1 * 1 向量
        global_feature = self.branch5_conv(global_feature)  # 修改通道数
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)  # 复制为特征图大小

        # 堆叠五个尺度信息
        feature_cat = torch.cat([conv_1, conv_3_1, conv_3_2, conv_3_3, global_feature], dim=1)  # 通道数为5 * dim_out
        result = self.conv_cat(feature_cat)  # 降维
        return result


class DeepLab(nn.Module):
    def __init__(self, num_classes):
        super(DeepLab, self).__init__()
        self.backbone = MobileNetV2()
        input_channel = 320  # 主干网络输出维度
        low_level_features = 24  # 低等级维度

        self.aspp = ASPP(dim_in=input_channel, dim_out= 256, rate=1)

        # 浅层特征升维
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_features, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        # 特征融合后进一步提取特征
        self.cat_conv = nn.Sequential(
            nn.Conv2d(48 + 256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Dropout(0.5),
        )

        # 分类卷积
        self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)
    
    def forward(self, x):
        H, W = x.size(2), x.size(3)  # 输入图片的高及宽
        
        # x bs, 320, 32, 32
        # low_level_features bs, 24, 128, 128
        low_level_features, x = self.backbone(x)

        x = self.aspp(x)  # bs, 256, 32, 32
        low_level_features = self.shortcut_conv(low_level_features)  # bs, 48, 128, 128

        # bs, 256, 128, 128
        x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear', align_corners=True)
        x = self.cat_conv(torch.cat((x, low_level_features), dim=1))  # bs, 256, 128, 128

        x = self.cls_conv(x)  # bs, num_class, 128, 128
        # bs, num_class, H, W
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x