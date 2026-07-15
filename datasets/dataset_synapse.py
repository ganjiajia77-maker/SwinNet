import os
import random

import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        # image: [3, H, W], label: [H, W]
        # 转换为 [H, W, 3] 和 [H, W] 便于处理
        image = image.transpose(1, 2, 0)  # [3, H, W] -> [H, W, 3]

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        
        h, w = image.shape[0], image.shape[1]  # image: [H, W, 3]
        if h != self.output_size[0] or w != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / h, self.output_size[1] / w, 1), order=3)
            label = zoom(label, (self.output_size[0] / h, self.output_size[1] / w), order=0)
        
        # 转换回 [3, H, W]
        image = image.transpose(2, 0, 1)  # [H, W, 3] -> [3, H, W]
        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split + '.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split in ["train", "val"] or self.sample_list[idx].strip('\n').split(",")[0].endswith(".npz"):
            slice_name = self.sample_list[idx].strip('\n').split(",")[0]
            if slice_name.endswith(".npz"):
                data_path = os.path.join(self.data_dir, slice_name)
            else:
                data_path = os.path.join(self.data_dir, slice_name + '.npz')
            data = np.load(data_path)
            try:
                image, label = data['image'], data['label']
            except:
                image, label = data['data'], data['seg']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample


class ImageDataset(Dataset):
    """用于加载JPG图像和PNG标签的数据集"""
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        
        # 获取所有image文件
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 读取 image - RGB 模式
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert('RGB')  # RGB 三通道
        image = np.array(image, dtype=np.float32) / 255.0  # 归一化到 [0, 1]
        
        # ImageNet 标准化
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
        image = image.transpose(2, 0, 1)  # [H, W, 3] -> [3, H, W]
        
        # 读取对应的 label
        label_name = image_name.replace('_sat.jpg', '_mask.png')
        label_path = os.path.join(self.label_dir, label_name)
        label = Image.open(label_path).convert('L')  # 灰度图
        label = np.array(label, dtype=np.float32)
        # 二值化: > 127 更安全（处理灰度边缘）
        label = (label > 127).astype(np.float32)
        
        sample = {'image': image, 'label': label}
        
        if self.transform:
            sample = self.transform(sample)
        
        sample['case_name'] = image_name.replace('_sat.jpg', '')
        return sample


class BinaryImageDataset(ImageDataset):
    """用于加载JPG/PNG数据，并按需指定样本文件名列表的数据集"""
    def __init__(self, image_dir, label_dir, image_files, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_files = sorted(image_files)
