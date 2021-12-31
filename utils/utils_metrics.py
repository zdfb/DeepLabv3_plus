import os
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F


###### 定义模型性能评判相关函数 ######


# F-score
def f_score(inputs, target, beta = 1, smooth = 1e-5, threshold = 0.5):
    n, c, h, w = inputs.size()  # 输入图像尺寸
    nt, ht, wt, ct = target.size()  # 标签尺寸

    # 尺寸不符合进行缩放
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    # 沿通道(num_class)进行softmax  
    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1)
    temp_target = target.view(n, -1, ct)  # 转化为 nt, ht * wt, c

    # 计算DICE系数
    # 将置信率大于0.5的部分置于1
    temp_inputs = torch.gt(temp_inputs, threshold).float()

    tp = torch.sum(temp_target[...,:-1] * temp_inputs, axis=[0,1])  # 计算tp
    fp = torch.sum(temp_inputs, axis=[0,1]) - tp  # 计算fp
    fn = torch.sum(temp_target[...,:-1], axis=[0,1]) - tp  # 计算fn

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)  # 计算F-score
    score = torch.mean(score)
    return score

# 下面这几段代码太tm秀了
# 计算混淆矩阵
def fast_hist(a, b, n):
    # a (h * w), b (h * w)
    k = (a >= 0) & (a < n)  # 剔除越界值

    # 计算混淆矩阵
    return np.bincount(n * a[k].astype(int) + b[k], minlength = n ** 2).reshape(n, n)

# 计算每个类别的IOU
def per_class_iou(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1)

# 计算每个类别的Recall
def per_class_PA_Recall(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1)

# 计算每个类别的精度
def per_class_Precision(hist):
    return np.diag(hist) / np.maximum(hist.sum(0), 1)

# 计算像素精度
def per_Accuracy(hist):
    return np.sum(np.diag(hist)) / np.maximum(np.sum(hist), 1)

# 计算mIOU值
def compute_mIOU(gt_dir, pred_dir, png_name_list, num_classes, name_classes):
    print('Num classes', num_classes)  # 类别数

    hist = np.zeros((num_classes, num_classes))  # 建立混淆矩阵 (num_class, num_class)
    
    # 获取预测图片与标签图片
    gt_imgs = [os.path.join(gt_dir, x + '.png') for x in png_name_list]
    pred_imgs = [os.path.join(pred_dir, x + '.png') for x in png_name_list]

    # 读取每一个 图片-标签对
    for index in range(len(gt_imgs)):
        # 将预测图片与标签图片均转化为numpy形式
        pred = np.array(Image.open(pred_imgs[index]))
        label = np.array(Image.open(gt_imgs[index]))
        
        # 若预测与标签尺寸不匹配则跳过
        if len(label.flatten()) != len(pred.flatten()):
            continue

        # 对每张图片计算hist矩阵
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
    
    IoUs = per_class_iou(hist)
    PA_Recall = per_class_PA_Recall(hist)
    Precision = per_class_Precision(hist)
    
    for ind_class in range(num_classes):
        print('===>' + name_classes[ind_class] + ':\tIou-' + str(round(IoUs[ind_class] * 100, 2)) \
            + '; Recall (equal to the PA)-' + str(round(PA_Recall[ind_class] * 100, 2))+ '; Precision-' + str(round(Precision[ind_class] * 100, 2)))

    print('===> mIoU: ' + str(round(np.nanmean(IoUs) * 100, 2)) + '; mPA: ' + str(round(np.nanmean(PA_Recall) * 100, 2)) + '; Accuracy: ' + str(round(per_Accuracy(hist) * 100, 2)))