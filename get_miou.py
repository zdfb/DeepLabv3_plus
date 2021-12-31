import os
import shutil
from PIL import Image
from tqdm import tqdm
from utils.utils import get_classes
from utils.utils_deeplab import DeepLabV3
from utils.utils_metrics import compute_mIOU


###### 计算模型mIOU ######


class Calculate_mIOU():
    def __init__(self):
        super(Calculate_mIOU, self).__init__()

        classes_path = 'model_data/name_classes.txt'  # 类别信息存放路径

        self.VOCdevkit_path = 'VOCdevkit'  # 数据集存储路径
        self.miou_out_path = 'miou_out'

        self.name_classes, self.num_classes = get_classes(classes_path)  # 获取标签名及数量

        # 类别名
        self.name_classes = ["background","aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
        # 取出测试集的图片id
        self.image_ids = open(os.path.join(self.VOCdevkit_path, "VOC2007/ImageSets/Segmentation/test.txt"), 'r').read().splitlines()
        # 存放真实标签的路径
        self.gt_dir = os.path.join(self.VOCdevkit_path, "VOC2007/SegmentationClass")
        # 存放预测结果的路径
        self.pred_dir = os.path.join(self.miou_out_path, 'segmentation_results')

        # 创建存储路径
        if not os.path.exists(self.pred_dir):
            os.makedirs(self.pred_dir)
    
    def calculate_mIOU(self):
        deeplab = DeepLabV3()  # 实例化Deeplabv3+模型

        for image_id in tqdm(self.image_ids):
            # 获得测试集中每个样本的路径
            image_path = os.path.join(self.VOCdevkit_path, "VOC2007/JPEGImages/" + image_id + ".jpg")
            image = Image.open(image_path)
            image = deeplab.get_miou_png(image)
            image.save(os.path.join(self.pred_dir, image_id + '.png'))
        
        compute_mIOU(self.gt_dir, self.pred_dir, self.image_ids, self.num_classes, self.name_classes)  # 计算各个评价指标
        shutil.rmtree(self.miou_out_path)

if __name__ == '__main__':
    miou = Calculate_mIOU()
    miou.calculate_mIOU()