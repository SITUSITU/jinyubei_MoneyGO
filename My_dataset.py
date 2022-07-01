import os, cv2
import torch.utils.data as data


import pandas as pd
from torchvision.io import read_image
from PIL import Image
import nibabel as nib
import numpy as np


class jinyubei_Dataset(data.Dataset):
    def __init__(self, imgs_dir_list, xlsx_file, img_transform=None):
        self.img_labels = pd.read_excel(xlsx_file)  # 读到的是dataframe格式
        self.img_labels = self.img_labels.set_index(keys="image_ID")
        self.imgs_dir_list = imgs_dir_list
        self.img_transform = img_transform

    def __len__(self):
        return len(self.imgs_dir_list)

    def __getitem__(self, idx):
        image_dir = self.imgs_dir_list[idx]
        image_name = image_dir.split("\\")[-1]
        image = Image.open(image_dir)
        label = self.img_labels.loc[image_name, "label"]
        """对于分类任务，一般只要处理数据图像"""
        if self.img_transform:
            image = self.img_transform(image)

        return image, label


def split_train_and_val(traval_list, nfold, select, seed=1):
    """
    传入训练集和验证集的数据路径文件名，根据则数和选择自动化分
    :param traval_list:
    :param nfold:
    :param select:
    :param seed:
    :return: 两个列表，每个列表储存的是数据的路径文件名
    """
    num_traval = len(traval_list)
    pid_idx = np.arange(num_traval)  # 获得顺序索引号
    np.random.seed(seed)
    np.random.shuffle(pid_idx)  # 打乱索引号
    n_fold_list = np.array_split(pid_idx, nfold)

    train_image_list = []
    val_image_list = []
    for i, fold in enumerate(n_fold_list):  # i代表份数，即第几则
        if i == select:  # 如果当前是所选择的则数
            for idx in fold:
                val_image_list.append(traval_list[idx])  # 就以当前则数里的乱序码，去获取总病人列表里对应的病人，储存到验证集中
        else:
            for idx in fold:
                train_image_list.append(traval_list[idx])

    return train_image_list, val_image_list