import cv2
import copy
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from nets.deeplabv3_plus import DeepLab
from utils.utils import cvtColor, preprocess_input, resize_image, ncolors, get_classes


###### 解析模型，生成最终结果 ######


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DeepLabV3(object):
    def __init__(self):
        super(DeepLabV3, self).__init__()

        model_path = 'model_data/deeplab.pth'  # 模型存储路径
        classes_path = 'model_data/name_classes.txt'  # 类别信息存放路径

        _, self.num_classes = get_classes(classes_path)
        self.input_shape = [512, 512]  # 输入图片的尺寸
        self.colors = [(0, 0, 0)] + ncolors(self.num_classes - 1)  # 背景 + 剩余随机颜色

        # 加载模型
        model = DeepLab(self.num_classes)
        # 加载训练权重
        model.load_state_dict(torch.load(model_path, map_location = device))
        model = model.eval()  # 推理模式

        self.model = model.to(device)
    
    def segmentate_image(self, image):
        image = cvtColor(image)  # 将输入图片转化为RGB形式

        old_image = copy.deepcopy(image)  # 将原图进行备份
        original_h = np.array(image).shape[0]  # 原图的高
        original_w = np.array(image).shape[1]  # 原图的宽

        # 不失真resize
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))

        #对输入图像进行预处理
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)  # 转化为tensor形式
            images = images.to(device)

            # 正向传播
            outputs = self.model(images)[0] # (num_class + 1, h, w)

            # 使用SoftMax取出每个点的种类
            # (num_class + 1, h, w) -> (h, w, num_class + 1)
            outputs = F.softmax(outputs.permute(1, 2, 0), dim = -1).cpu().numpy()
            # 截取灰条部分
            outputs = outputs[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                    int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
            
            # 将输出还原至原来的尺寸
            outputs = cv2.resize(outputs, (original_w, original_h), interpolation = cv2.INTER_LINEAR)

            # 取出每个像素点的种类
            # (h, w, num_class + 1) -> (h, w)
            outputs = outputs.argmax(axis = -1)
        
        seg_img = np.zeros((np.shape(outputs)[0], np.shape(outputs)[1], 3))
        # 创建新图，根据预测类别赋予颜色
        for c in range(self.num_classes):
            seg_img[:,:,0] += ((outputs[:,: ] == c )*( self.colors[c][0] )).astype('uint8')
            seg_img[:,:,1] += ((outputs[:,: ] == c )*( self.colors[c][1] )).astype('uint8')
            seg_img[:,:,2] += ((outputs[:,: ] == c )*( self.colors[c][2] )).astype('uint8')
        
        # 将该图片转化为Image形式
        image = Image.fromarray(np.uint8(seg_img))
        # 混合原图与预测后图片
        image = Image.blend(old_image, image, 0.7)
        return image
    
    def get_miou_png(self, image):
        image = cvtColor(image)  # 将输入图片转化为RGB形式

        original_h = np.array(image).shape[0]  # 原图的高
        original_w = np.array(image).shape[1]  # 原图的宽

        # 不失真resize
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))

        #对输入图像进行预处理
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)
        with torch.no_grad():
            images = torch.from_numpy(image_data)  # 转化为tensor形式
            images = images.to(device)

            # 正向传播
            outputs = self.model(images)[0] # (num_class + 1, h, w)

            # 使用SoftMax取出每个点的种类
            # (num_class + 1, h, w) -> (h, w, num_class + 1)
            outputs = F.softmax(outputs.permute(1, 2, 0), dim = -1).cpu().numpy()
            # 截取灰条部分
            outputs = outputs[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                    int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
            
            # 将输出还原至原来的尺寸
            outputs = cv2.resize(outputs, (original_w, original_h), interpolation = cv2.INTER_LINEAR)

            # 取出每个像素点的种类
            # (h, w, num_class + 1) -> (h, w)
            outputs = outputs.argmax(axis = -1)
        image = Image.fromarray(np.uint8(outputs))
        return image