import os
import cv2
import torch
import pickle
import random
import numbers
import numpy as np
from torch.utils import data
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image


class ImageDataTrain(data.Dataset):
    def __init__(self, data_root, data_list):
        self.sal_root = data_root
        self.sal_source = data_list

        with open(self.sal_source, 'r') as f:
            self.sal_list = [x.strip() for x in f.readlines()]

        self.sal_num = len(self.sal_list)

    def __getitem__(self, item):
        # sal data loading
        im_name = self.sal_list[item % self.sal_num].split()[1]
        gt_name = self.sal_list[item % self.sal_num].split()[2]
        edge_name = self.sal_list[item % self.sal_num].split()[3]
        sal_image = load_image(os.path.join(self.sal_root, im_name))
        sal_label = load_sal_label(os.path.join(self.sal_root, gt_name))
        sal_edge = load_sal_label(os.path.join(self.sal_root, edge_name))
        sal_image = torch.Tensor(sal_image)
        sal_label = torch.Tensor(sal_label)
        sal_edge = torch.Tensor(sal_edge)

        sample = {'sal_image': sal_image, 'sal_label': sal_label, 'sal_edge': sal_edge}
        return sample

    def __len__(self):
        return self.sal_num

class ImageDataTest(data.Dataset):
    def __init__(self, data_root, data_list):
        self.data_root = data_root
        self.data_list = data_list
        with open(self.data_list, 'r') as f:
            self.image_list = [x.strip() for x in f.readlines()]

        self.image_num = len(self.image_list)

    def __getitem__(self, item):
        name = self.image_list[item % self.image_num].split()[0]
        im_name = self.image_list[item % self.image_num].split()[1]
        image, im_size = load_image_test(os.path.join(self.data_root, im_name))
        image = torch.Tensor(image)

        return {'image': image, 'name': name, 'size': im_size}

    def __len__(self):
        return self.image_num


def get_loader(config, mode='train', pin=True):
    shuffle = False
    if mode == 'train':
        shuffle = True
        dataset = ImageDataTrain(config.train_root, config.train_list)
        data_loader = data.DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=shuffle, num_workers=config.num_thread, pin_memory=pin)
    else:
        dataset = ImageDataTest(config.test_root, config.test_list)
        data_loader = data.DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=shuffle, num_workers=config.num_thread, pin_memory=pin)
    return data_loader

def load_image(path):
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    im = cv2.imread(path) #B G R
    in_ = np.array(im, dtype=np.float32)
    in_ = (in_ - np.array((118.601771, 116.073331, 110.343461))) / np.array((74.555715, 69.207232, 70.965941))
    in_ = cv2.resize(in_, dsize=(1024, 512), interpolation=cv2.INTER_LINEAR)
    in_ = in_.transpose((2,0,1))
    return in_

def load_image_test(path):
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    im = cv2.imread(path)
    in_ = np.array(im, dtype=np.float32)
    im_size = tuple(in_.shape[:2])
    in_ = (in_ - np.array((118.601771, 116.073331, 110.343461))) / np.array((74.555715, 69.207232, 70.965941))
    in_ = cv2.resize(in_, dsize=(1024, 512), interpolation=cv2.INTER_LINEAR)
    in_ = in_.transpose((2,0,1))
    return in_, im_size

def load_sal_label(path):
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    im = Image.open(path)
    label = np.array(im, dtype=np.float32)
    if len(label.shape) == 3:
        label = label[:,:,0]
    label = label / 255.
    label = cv2.resize(label, dsize=(1024, 512), interpolation=cv2.INTER_NEAREST)
    label = label[np.newaxis, ...]
    return label

def load_sal_label_test(path):
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    im = Image.open(path)
    label = np.array(im, dtype=np.float32)
    if len(label.shape) == 3:
        label = label[:,:,0]
    label = label / 255.
    label = cv2.resize(label, dsize=(1024, 512), interpolation=cv2.INTER_NEAREST)
    label = label[np.newaxis, ...]
    return label

