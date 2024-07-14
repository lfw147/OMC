import os
import pandas as pd
import torch.utils.data as data
from utils import *
from torchvision import transforms

class RafDataset(data.Dataset):
    def __init__(self,images_path,labels,stage,basic_aug=True, transform=None):
        self.images_path=images_path
        self.labels=labels
        self.stage=stage
        self.basic_aug=basic_aug
        self.transform=transform
        # 数据增强函数,翻转图像并添加高斯噪声
        self.aug_func = [flip_image, add_g]

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        # 获取标签
        label = self.labels[idx]
        # 读取图像
        image = cv2.imread(self.images_path[idx])
        # 转换图像通道顺序（BGR to RGB）
        image = image[:, :, ::-1]

        # if not self.clean:
        #     image1 = image
        #     # 对图像应用增强函数
        #     image1 = self.aug_func[0](image)
        #     # 对图像应用变换
        #     image1 = self.transform(image1)

        # 如果处于训练阶段，应用数据增强
        if self.stage == 'train':
            if self.basic_aug and random.uniform(0, 1) > 0.5:
                image = self.aug_func[1](image)

        # 应用图像变换
        if self.transform is not None:
            image = self.transform(image)
        # #如果self.clean为True，那么会对原始图像image执行概率为1的水平翻转操作（即一定执行翻转）
        # if self.clean:
        #     image1 = transforms.RandomHorizontalFlip(p=1)(image)

        return image, label