"""import 一些自定义与处理方式所需要用到的包"""
import torch
import numpy as np
from torchvision.transforms import functional as F

class XXX_preset(object):
    def __init__(self):
        """在这里把传进来的参数monument传给self"""
        """如果该预处理不需要参数，则可以省略init，直接写call"""
    def __call__(self, image, target):
        """

        在call函数里，利用self获得的参数和import进来的包
        对传进来的图像和标签进行预处理
        并返回处理之后的图像与标签
        """
        return image, target


class YYY_preset():

    """同理写出其余的各种与处理方式"""
    pass


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)  # 把图像从0-255缩放至0-1
        target = torch.as_tensor(np.array(target), dtype=torch.int64)  # 标签不能缩放，因为其数值代表了类别索引
        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


"""最后在train文件中，import当前的文件，并调用文件.预处理方式即可"""


