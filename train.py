import torch
import numpy as np
import torch.optim as optim
from utils.utils import get_classes
import torch.backends.cudnn as cudnn
from nets.deeplabv3_plus import DeepLab
from torch.utils.data import DataLoader
from utils.utils_fit import fit_one_epoch
from utils.dataloader import DeeplabDataset, deeplab_dataset_collate


###### 训练模型 ######


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class train_deeplabv3():
    def __init__(self):
        super(train_deeplabv3, self).__init__()

        model_path = 'model_data/deeplab.pth'  # 模型保存路径
        classes_path = 'model_data/name_classes.txt'  # 类别信息存放路径

        train_annotation_path = 'VOCdevkit/VOC2007/ImageSets/Segmentation/train.txt'  # 训练集样本图片存储路径
        test_annotation_path = 'VOCdevkit/VOC2007/ImageSets/Segmentation/test.txt'  # 测试集样本图片存储路径
        
        self.VOC_path = 'VOCdevkit'
        _, self.num_classes = get_classes(classes_path)
        self.input_shape = [512, 512]  # 输入尺寸

        self.dice_loss = False  # 是否使用Dice_loss
        self.focal_loss = False  # 是否使用Focal_loss

        self.class_weights = np.ones([self.num_classes], np.float32)  # 默认使用平衡权重

        # 创建DeeplabV3plus模型
        model = DeepLab(self.num_classes)
        print('Load Weights from {}.'.format(model_path))

        model_dict = model.state_dict()  # 模型参数
        pretrained_dict = torch.load(model_path, map_location = device)  # 预训练模型的参数

        # 替换key相同且shape相同的值
        pretrained_dict = {k:v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)  # 更新参数
        model.load_state_dict(model_dict)  # 加载参数

        if torch.cuda.is_available():
            cudnn.benchmark = True
            model = model.to(device)
        
        self.model = model

        with open(train_annotation_path, 'r') as f:
            self.train_lines = f.readlines()  # 读取训练集数据
        with open(test_annotation_path, 'r') as f:
            self.test_lines = f.readlines()  # 读取测试集数据

        self.loss_test_min = 1e9  # 初始化最小测试集loss

    def train(self, batch_size, learning_rate, start_epoch, end_epoch, Freeze = False):

        # 定义优化器
        optimizer = optim.Adam(self.model.parameters(), learning_rate, weight_decay = 5e-4)

        # 学习率下降策略
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.94)

        # 定义训练集与测试集
        train_dataset = DeeplabDataset(self.train_lines, self.input_shape, self.num_classes, train = True, dataset_path = self.VOC_path)
        test_dataset = DeeplabDataset(self.test_lines, self.input_shape, self.num_classes, train = False, dataset_path = self.VOC_path)
        train_data = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = 4,
                                  pin_memory = True, drop_last = True, collate_fn = deeplab_dataset_collate)
        test_data = DataLoader(test_dataset, shuffle = True, batch_size = batch_size,num_workers = 4,
                                  pin_memory = True, drop_last = True, collate_fn = deeplab_dataset_collate)
        
        # 冻结backbone参数
        if Freeze:
            for param in self.model.backbone.parameters():
                param.requires_grad = False
        else:
            for param in self.model.backbone.parameters():
                param.requires_grad = True
        

        # 开始训练
        for epoch in range(start_epoch, end_epoch):
            print('Epoch: ',epoch)
            train_loss, test_loss = fit_one_epoch(self.model, optimizer, train_data, test_data, self.dice_loss, self.focal_loss, self.class_weights, self.num_classes, device)
            lr_scheduler.step()

            # 若测试集loss小于当前极小值， 保存当前模型
            if test_loss < self.loss_test_min:
                self.loss_test_min = test_loss
                torch.save(self.model.state_dict(), 'deeplab.pth')
    

    def total_train(self):

        # 首先进行backbone冻结训练
        Freeze_batch_size = 8
        Freeze_lr = 5e-4
        Init_epoch = 0
        Freeze_epoch = 50

        self.train(Freeze_batch_size, Freeze_lr, Init_epoch, Freeze_epoch, Freeze = True)

        # 解冻backbone训练
        batch_size = 4
        learning_rate = 5e-5
        end_epoch = 100
        self.train(batch_size, learning_rate, Freeze_epoch, end_epoch, Freeze = False)

if __name__ == "__main__":
    train = train_deeplabv3()
    train.total_train()