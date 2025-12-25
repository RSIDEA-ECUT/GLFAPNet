import os
import cv2
import numpy as np
import torch
from osgeo import gdal
from torch.utils.data import Dataset
from torchvision.transforms import transforms


# 训练数据集
class TrainDataset(Dataset):
    # 初始化，一般读取数据的路径
    def __init__(self, mul_root_dir, pan_root_dir):
        self.mul_root_dir = mul_root_dir
        self.pan_root_dir = pan_root_dir
        self.mul_img_path = os.listdir(self.mul_root_dir)
        self.pan_img_path = os.listdir(self.pan_root_dir)

    # 读取数据信息，一般返回读取的数据和标签
    def __getitem__(self, idx):
        mul_img_name = self.mul_img_path[idx]
        pan_img_name = self.pan_img_path[idx]

        mul_img_idx_path = os.path.join(self.mul_root_dir, mul_img_name)
        pan_img_idx_path = os.path.join(self.pan_root_dir, pan_img_name)
        trans_tensor = transforms.ToTensor()

        # 处理多光谱图像，多光谱数据先进行4倍下采样，再进行4倍上采样,最后尺寸大小32*32
        mul_data = gdal.Open(mul_img_idx_path)
        mul_array = np.float32(mul_data.ReadAsArray().transpose(1, 2, 0))

        down_mul_img = cv2.resize(mul_array, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
        up_mul_img = cv2.resize(down_mul_img, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        up_mul_img[up_mul_img < 0] = 0
        up_mul_img[up_mul_img > 255] = 255

        input_mul_img = trans_tensor(np.uint8(up_mul_img))

        # 处理全色图像，全色影像进行4倍下采样，和多光谱影像具有一样的尺寸大小32*32
        pan_data = gdal.Open(pan_img_idx_path)
        pan_array = np.float32(pan_data.ReadAsArray())

        down_pan_img = cv2.resize(pan_array, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
        down_pan_img[down_pan_img < 0] = 0
        down_pan_img[down_pan_img > 255] = 255

        input_pan_img = trans_tensor(np.uint8(down_pan_img))

        # 最后输入模型的为：4波段多光谱影像和1波段全色影像拼接的5波段数据-->(B,G,R,NIR,PAN),标签为原始4波段的多光谱影像
        input_img = torch.cat([input_mul_img, input_pan_img], dim=0)
        label = trans_tensor(np.uint8(mul_array))
        return input_img, label

    # 返回数据集的长度
    def __len__(self):
        return len(self.mul_img_path)


# 真实测试数据集
class RealTestDataset(Dataset):
    # 初始化，一般读取数据的路径
    def __init__(self, mul_root_dir, pan_root_dir):
        self.mul_root_dir = mul_root_dir
        self.pan_root_dir = pan_root_dir
        self.mul_img_path = os.listdir(self.mul_root_dir)
        self.pan_img_path = os.listdir(self.pan_root_dir)

    # 读取数据信息，一般返回读取的数据和标签
    def __getitem__(self, idx):
        mul_img_name = self.mul_img_path[idx]
        pan_img_name = self.pan_img_path[idx]

        mul_img_idx_path = os.path.join(self.mul_root_dir, mul_img_name)
        pan_img_idx_path = os.path.join(self.pan_root_dir, pan_img_name)

        trans_tensor = transforms.ToTensor()

        # 处理多光谱图像
        mul_data = gdal.Open(mul_img_idx_path)
        mul_array = np.float32(mul_data.ReadAsArray().transpose(1, 2, 0))

        # 多光谱数据直接进行四倍上采样
        up_mul_img = cv2.resize(np.float32(mul_array), (0, 0), fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        up_mul_img[up_mul_img < 0] = 0
        up_mul_img[up_mul_img > 255] = 255
        input_mul_img = trans_tensor(np.uint8(up_mul_img))

        # 处理全色图像
        pan_data = gdal.Open(pan_img_idx_path)
        pan_array = pan_data.ReadAsArray()
        input_pan_img = trans_tensor(np.uint8(pan_array))

        # 最后输入模型的为：4波段多光谱影像和1波段全色影像拼接的5波段数据-->(B,G,R,NIR,PAN)
        input_img = torch.cat([input_mul_img, input_pan_img], dim=0)
        mul_array = trans_tensor(np.uint8(mul_array))
        return input_img, input_pan_img, mul_array

    # 返回数据集的长度
    def __len__(self):
        return len(self.mul_img_path)
