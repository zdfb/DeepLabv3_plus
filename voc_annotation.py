import os
import random


###### 读取及分割数据集 ######


# VOC形式数据集路径
VOCdevkit_path = 'VOCdevkit'

# 训练集:测试集 = 9:1
train_percent = 0.9

# 生成数据集划分txt文件
def generate_split_txt():

    # 标签存储路径
    segfilepath = os.path.join(VOCdevkit_path, 'VOC2007/SegmentationClass')
    # 生成的训练集与测试集id存储路径
    saveBasePath = os.path.join(VOCdevkit_path, 'VOC2007/ImageSets/Segmentation')

    # 遍历路径下所有文件
    total_seg = os.listdir(segfilepath)
    # 所有样本总数
    num_samples = len(total_seg)
    # 随机打乱样本list
    random.shuffle(total_seg)
    # 训练样本总数
    train_num = int(train_percent * num_samples)

    # 创建存储训练及测试样本id的txt文件
    f_train = open(os.path.join(saveBasePath, 'train.txt'), 'w')
    f_test = open(os.path.join(saveBasePath, 'test.txt'), 'w')

    for index, seg in enumerate(total_seg):
        # 剔除.png后缀
        name = seg[:-4] + '\n'
        # 判断是否处于训练样本内
        if index < train_num:
            f_train.write(name)  # 写入训练集txt文件
        else:
            f_test.write(name)  # 写入测试集txt文件
    
    # 关闭打开的txt文件
    f_train.close()
    f_test.close()
    print("Generate txts in ImageSets done.")


if __name__ == "__main__":
    # 设置随机种
    random.seed(0)
    generate_split_txt()