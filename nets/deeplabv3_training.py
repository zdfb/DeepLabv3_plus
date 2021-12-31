import torch
import torch.nn as nn
import torch.nn.functional as F


###### 定义损失函数 ######


# 定义Crossentropy Loss
def CE_Loss(inputs, target, cls_weights, num_classes = 21):
    n, c, h, w = inputs.size()  # 获取输入的尺寸
    nt, ht, wt = target.size()  # 获取标签的尺寸
    
    # 尺寸不符合进行放大
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, (ht, wt), mode='bilinear', align_corners=True)
    
    # n, c, h, w -> n, h, c, w -> n, h, w, c -> n * h * w, c
    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)  # nt * ht * wt

    # cls_weights  对crossentropy进行加权
    # ignore_index  设置为第22类，即不易区分的边界
    loss = nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes)(temp_inputs, temp_target)
    return loss

# 定义Focalloss
def Focal_loss(inputs, target, cls_weights, num_classes = 21, alpha = 0.5, gamma = 2):
    n, c, h, w = inputs.size()  # 获取输入的尺寸
    nt, ht, wt = target.size()  # 获取标签的尺寸
    
    # 尺寸不符合进行放大
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, (ht, wt), mode='bilinear', align_corners=True)
    
    # n, c, h, w -> n, h, c, w -> n, h, w, c -> n * h * w, c
    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)  # nt * ht * w
    
    # 不进行平均值计算
    logpt = -nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes,reduction='none')(temp_inputs, temp_target)

    pt = torch.exp(logpt)  # 取得对应样本的预测值

    if alpha is not None:
        logpt *= alpha  # loss加权
    
    loss = -((1 - pt) ** gamma) * logpt
    loss = loss.mean()
    return loss


# 定义Dice_loss
def Dice_loss(inputs, target, beta = 1, smooth = 1e-5):
    n, c, h, w = inputs.size()  # 获取输入的尺寸
    nt, ht, wt, ct = target.size()  # 获取标签的尺寸
    
    # 尺寸不符合进行放大
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, (ht, wt), mode='bilinear', align_corners=True)
    
    # 对每个通道计算softmax
    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
    temp_target = target.view(n, -1, ct)

    tp = torch.sum(temp_target[...,:-1] * temp_inputs, axis=[0,1])  # 计算TP 真正例
    fp = torch.sum(temp_inputs, axis=[0,1]) - tp  # 计算FP 假正例
    fn = torch.sum(temp_target[...,:-1], axis=[0,1]) - tp  # 计算FN 假负例

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    dice_loss = 1 - torch.mean(score)
    return dice_loss