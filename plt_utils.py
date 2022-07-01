import matplotlib.pyplot as plt
import os
import numpy as np


def plt_xOy(x_ndarray,
            y_ndarray: list,
            args,
            png_name,
            figure_num,
            x_axis_label=None,
            y_axis_label=None,
            legend: list = None):
    """
    传入画图所需要的变量，自动画图并保存到目标路径

    :param x_ndarray: 一维数组，表示 X 轴取值
    :param y_ndarray: 以【】列表形式保存的多个一维数组，表示多个函数在 Y 轴的取值
    :param args:
    :param png_name: 当前画出来的图片以什么名字保存
    :param figure_num: 当前画的是第几张图片
    :param x_axis_label:
    :param y_axis_label:
    :param legend: 以【】列表形式保存的多个字符串，分别与多个函数的函数名一一对应
    :return:
    """
    plt.figure(num=figure_num)
    for i in range(len(y_ndarray)):
        plt.plot(x_ndarray, y_ndarray[i])  # plot(横坐标的取值，纵坐标的取值)

    plt.xlabel(x_axis_label)  # 不能用中文
    plt.ylabel(y_axis_label)
    if legend is not None:
        plt.legend(legend)

    plt.savefig(os.path.join(args.model_save_path, png_name))


if __name__ == "__main__":
    def sigmoid(x):
        s = 1/(1+np.exp(-x))
        return s
    x = np.arange(start=-5, stop=5, step=0.1)
    plt_xOy(x, [sigmoid(x), sigmoid(np.log(np.abs(x)))], legend=["sigmoid(x)", "sigmoid(log(abs(x)))"])
    plt.show()