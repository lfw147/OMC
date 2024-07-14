import torch
import cv2
import numpy as np
import random

# 添加高斯噪声
def add_g(image_array, mean=0.0, var=30):
    # 计算标准差
    std = var ** 0.5
    # 向图像添加高斯噪声
    image_add = image_array + np.random.normal(mean, std, image_array.shape)
    # 裁剪值并转换为无符号8位整数类型
    image_add = np.clip(image_add, 0, 255).astype(np.uint8)
    return image_add

# 水平翻转图像
def flip_image(image_array):
    return cv2.flip(image_array, 1)

# 设置随机种子，以确保实验的可重复性
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# 生成用于翻转注意力映射的网格
# def generate_flip_grid(w, h, device):
#     # used to flip attention maps
#     # 生成宽度为w的一维张量，然后将其扩展为维度为(h, w)的二维张量
#     x_ = torch.arange(w).view(1, -1).expand(h, -1)
#     # 生成高度为h的一维张量，然后将其扩展为维度为(h, w)的二维张
#     y_ = torch.arange(h).view(-1, 1).expand(-1, w)
#     # 将x_和y_沿dim=0堆叠，然后将其转换为float类型并移至指定设备（CPU或GPU
#     grid = torch.stack([x_, y_], dim=0).float().to(device)
#     # 在dim=0处添加一个新维度，并扩展该维度以使其具有维度(1, 2, h, w)
#     grid = grid.unsqueeze(0).expand(1, -1, -1, -1)
#     # 将x轴坐标归一化到[-1, 1]范围
#     grid[:, 0, :, :] = 2 * grid[:, 0, :, :] / (w - 1) - 1
#     # 将y轴坐标归一化到[-1, 1]范围
#     grid[:, 1, :, :] = 2 * grid[:, 1, :, :] / (h - 1) - 1
#     # 翻转x轴坐标
#     grid[:, 0, :, :] = -grid[:, 0, :, :]
#     return grid