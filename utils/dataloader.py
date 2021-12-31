import os
import cv2
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from utils.utils import preprocess_input, cvtColor


###### 定义数据读取 ######


# 定义数据读取类
class DeeplabDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path):
        super(DeeplabDataset, self).__init__()
        self.annotation_lines = annotation_lines  # 标签信息行
        self.length = len(annotation_lines)  # 标签信息行总数
        self.input_shape = input_shape  # 要求的图像尺寸
        self.num_classes = num_classes  # 预测类别数
        self.train = train
        self.dataset_path = dataset_path  # 数据集路径
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]  
        name = annotation_line.split()[0]  # 取得当前样本的文件名

        # 读取输入图片
        jpg = Image.open(os.path.join(os.path.join(self.dataset_path, "VOC2007/JPEGImages"), name + ".jpg"))
        # 读取标签图片
        png = Image.open(os.path.join(os.path.join(self.dataset_path, "VOC2007/SegmentationClass"), name + ".png"))

        # 进行数据增强
        jpg, png = self.get_random_data(jpg, png, self.input_shape, random = self.train)
        
        # 由h, w, c -> c, h, w
        jpg = np.transpose(preprocess_input(np.array(jpg, np.float64)), [2, 0, 1])
        png = np.array(png)  # 标签转化为numpy形式

        png[png >= self.num_classes] = self.num_classes  # 防止像素值越界

        # 转化为one_hot形式
        # n, w, 1 -> n, w, num_class + 2  # 0 为背景类，最后一维是不易区分的边界
        seg_labels  = np.eye(self.num_classes + 1)[png.reshape([-1])]
        seg_labels = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))

        return jpg, png, seg_labels

    
    # 产生a～b范围内的随机数
    def rand(self, a = 0, b = 1):
        return np.random.rand() * (b - a) + a
    
    # 进行数据增强
    def get_random_data(self, image, label, input_shape, jitter = .3, hue = .1, sat = 1.5, val = 1.5, random=True):
        image = cvtColor(image)  # 转化为RGB格式
        label = Image.fromarray(np.array(label))  # 转化为图片
        h, w = input_shape  # 要求的高与宽
        
        # 处于测试模式
        if not random:
            iw, ih = image.size  # 输入图像的尺寸
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)

            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', [w, h], (128, 128, 128))  # 背景用灰色填充
            new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

            label = label.resize((nw, nh), Image.NEAREST)
            new_label = Image.new('L', [w, h], (0))  # 背景用黑色填充
            new_label.paste(label, ((w - nw) // 2, (h - nh) // 2))
            return new_image, new_label
        
        # 处于训练模式

        # 对图像进行随机缩放
        rand_jit1 = self.rand(1 - jitter,1 + jitter)
        rand_jit2 = self.rand(1 - jitter,1 + jitter)
        new_ar = w / h * rand_jit1 / rand_jit2

        scale = self.rand(0.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)

        image = image.resize((nw, nh), Image.BICUBIC)
        label = label.resize((nw, nh), Image.NEAREST)
        
        # 对图像进行随机左右翻转
        flip = self.rand() < .5
        if flip: 
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        
        # 随机将图像放置在背景中
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w,h), (128, 128, 128))  # 输入图像用灰色填充背景
        new_label = Image.new('L', (w, h), (0))  # 标签用黑色填充背景
        new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))
        image = new_image
        label = new_label

        # 进行随机色域变换
        hue = self.rand(-hue, hue)
        sat = self.rand(1, sat) if self.rand()<.5 else 1 / self.rand(1, sat)
        val = self.rand(1, val) if self.rand()<.5 else 1 / self.rand(1, val)
        x = cv2.cvtColor(np.array(image,np.float32) / 255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue*360
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:,:, 0] > 360, 0] = 360
        x[:, :, 1:][x[:, :, 1:] > 1] = 1
        x[x < 0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255

        return image_data,label


# 定义数据加载时按照batch的堆叠方式
def deeplab_dataset_collate(batch):
    images = []
    pngs = []
    seg_labels = []
    for img, png, labels in batch:
        images.append(img)
        pngs.append(png)
        seg_labels.append(labels)
    images = np.array(images)
    pngs = np.array(pngs)
    seg_labels = np.array(seg_labels)
    return images, pngs, seg_labels       