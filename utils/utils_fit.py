import time
import torch
import numpy as np
from utils.utils_metrics import f_score
from nets.deeplabv3_training import CE_Loss, Dice_loss, Focal_loss


###### Deeplabv3模型训练一个epoch ######


def fit_one_epoch(model, optimizer, train_data, test_data, dice_loss, focal_loss, cls_weights, num_classes, device):

    start_time = time.time()  # 获取当前时间
    model.train()  # 训练模式

    loss_train_list = []
    fscore_train_list = []
    for step, data in enumerate(train_data):
        imgs, pngs, labels = data  # 取出输入图片，标签图片及用于DICE_loss的one-hot形式标签图片

        # 将数据转化为torch.tensor形式
        with torch.no_grad():
            imgs = torch.from_numpy(imgs).type(torch.FloatTensor).to(device)
            pngs = torch.from_numpy(pngs).long().to(device)
            labels = torch.from_numpy(labels).long().to(device)
            weights = torch.from_numpy(cls_weights).to(device)
        
        optimizer.zero_grad()  # 清零梯度
        outputs = model(imgs)  # 前向传播

        # 是否使用Focal_loss
        if focal_loss:
            loss = Focal_loss(outputs, pngs, weights, num_classes = num_classes)
        else:
            loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)
        
        # 是否使用DICE_loss
        if dice_loss:
            main_dice = Dice_loss(outputs, labels)
            loss = loss + main_dice
        
        with torch.no_grad():
            score = f_score(outputs, labels)
        
        loss.backward()  # 反向传播
        optimizer.step()  # 优化器迭代

        loss_train_list.append(loss.item())
        fscore_train_list.append(score.item())

        # 画进度条
        rate = (step + 1) / len(train_data)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss, f_score: {:^3.0f}%[{}->{}]{:.3f}, {:.3f}".format(int(rate * 100), a, b, loss, score), end="")
    print() 

    model.eval()  # 测试模式

    loss_test_list = []
    fscore_test_list = []
    for step, data in enumerate(test_data):
        imgs, pngs, labels = data  # 取出输入图片，标签图片及用于DICE_loss的one-hot形式标签图片
        with torch.no_grad():
            imgs = torch.from_numpy(imgs).type(torch.FloatTensor).to(device)
            pngs = torch.from_numpy(pngs).long().to(device)
            labels = torch.from_numpy(labels).long().to(device)
            weights = torch.from_numpy(cls_weights).to(device)
        
        outputs = model(imgs)
        if focal_loss:
            loss = Focal_loss(outputs, pngs, weights, num_classes = num_classes)
        else:
            loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)
        
        # 是否使用DICE_loss
        if dice_loss:
            main_dice = Dice_loss(outputs, labels)
            loss = loss + main_dice
        
        with torch.no_grad():
            score = f_score(outputs, labels)
        
        loss_test_list.append(loss.item())
        fscore_test_list.append(score.item())

        # 画进度条
        rate = (step + 1) / len(test_data)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtest loss, f_score: {:^3.0f}%[{}->{}]{:.3f}, {:.3f}".format(int(rate * 100), a, b, loss, score), end="")
    print()

    train_loss = np.mean(loss_train_list)  # 该epoch总的训练loss
    test_loss = np.mean(loss_test_list)  # 该epoch总的测试loss

    train_fscore = np.mean(fscore_train_list)  # 该epoch总的训练fscore
    test_fscore = np.mean(fscore_test_list)  # 该epoch总的测试fscore

    stop_time = time.time()  # 获取当前时间
    
    print('total_train_loss: %.3f, total_test_loss: %.3f, total_train_fscore:%.3f, total_test_fscore:%.3f, epoch_time: %.3f.'%(train_loss, test_loss, train_fscore, test_fscore, stop_time - start_time))
    return train_loss, test_loss