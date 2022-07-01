import torch
import pretrainedmodels
from torch import nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import os, glob
import numpy as np

from log_utils import get_logger
import train_val_utils
from plt_utils import plt_xOy
from My_dataset import split_train_and_val, jinyubei_Dataset


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="一段描述")
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--data_root", default="./new_dataset", type=str)
    parser.add_argument("--in_channel", default=3, type=int)
    parser.add_argument("--num_classes", default=4, type=int)
    parser.add_argument("--pre_train", default=None, type=str)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--epochs", default=500, type=int)
    parser.add_argument("--nfold", default=5, type=int)

    parser.add_argument("--select_fold", default=2, type=int)
    parser.add_argument('--resume', default='', help='断点续训所需要的文件路径')
    parser.add_argument('--start_epoch', default=-1, type=int, metavar='N',
                        help='从哪个epoch开始。断点续训时会被断点文件读到的epoch重新定义')
    parser.add_argument("--logger_path", default="./log/启动动态学习率.txt", type=str, help="传入日志txt文件的路径文件名。没有则自动创建")
    parser.add_argument("--model_save_path", default="./model_weight/启动动态学习率", type=str, help="传入模型权重保存路径，没有则自动创建")
    args = parser.parse_args()

    return args


def main(args):
    logger = get_logger(args.logger_path)  # 生成日志记录器
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"当前使用设备为{device}")  # 记录日志
    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])

    """=========================================获取dataset并预处理打包===================================================="""
    data_transform = {  # 这里把训练集和验证集的一系列预处理各自打包好并放进一个字典中
        "train": transforms.Compose([transforms.RandomResizedCrop((224, 224)),  # 随机裁剪到224*224大小
                                     transforms.RandomHorizontalFlip(),  # 随机翻转
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    All_images_list = glob.glob(os.path.join(args.data_root, "image", "*"))
    train_image_list, val_image_list = split_train_and_val(traval_list=All_images_list, nfold=args.nfold,
                                                           select=args.select_fold)
    train_dataset = jinyubei_Dataset(imgs_dir_list=train_image_list, img_transform=data_transform["train"],
                                     xlsx_file=os.path.join(args.data_root, "All_image_name_and_class.xlsx"))
    val_dataset = jinyubei_Dataset(imgs_dir_list=val_image_list, img_transform=data_transform["val"],
                                   xlsx_file=os.path.join(args.data_root, "All_image_name_and_class.xlsx"))

    logger.info(f"总共{len(All_images_list)}个数据，当前采用{args.nfold}则交叉验证，采用第{args.select_fold}则。"
                f"训练集中共{len(train_dataset)}例。验证集共{len(val_dataset)}例")

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=nw,
                              )

    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=nw,
                            )

    """=========================================建立网络并设置超参数===================================================="""
    model = pretrainedmodels.__dict__['xception'](num_classes=args.num_classes, pretrained=None).to(device)
    if args.pre_train is not None:
        assert os.path.isfile(args.pre_train), "预训练文件不存在"
        model.load_state_dict(torch.load(args.pre_train, map_location=device))  # 迁移学习
        logger.info(f"成功载入预训练权重：{args.pre_train}")

    loss_function = nn.CrossEntropyLoss()  # 设置损失函数
    optimizer = optim.Adam(params=model.parameters(),
                           lr=args.lr)  # 选用优化器，并传入需要训练的参数和默认学习率

    # 每隔 10 个epoch 将学习率乘以0.9
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.9, last_epoch=args.start_epoch)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")  # 先把模型读到内存。这里的Checkpoint是一个字典
        model.load_state_dict(checkpoint["model"])  # 通过键索引获取权重字典
        optimizer.load_state_dict(checkpoint["optimizer"])  # 通过键索引获取优化器字典
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])  # 通过键索引获取动态学习率字典
        args.start_epoch = checkpoint['epoch'] + 1

    """=========================================训练与验证===================================================="""
    if not os.path.exists(args.model_save_path):
        os.mkdir(args.model_save_path)

    # 创建画图所需要用的所有空数组
    epochs_plt = np.array([])
    train_loss_plt = np.array([])
    val_loss_plt = np.array([])
    acc_plt = np.array([])
    # 定义以某个指标为标准，保存最好的模型
    best_metric = 0.0

    for epoch in range(args.epochs):  # epoch循环，每个循环都会用完所有的训练数据

        """===========================训练并验证一个epoch，两个函数返回哪些指标可自行改写======================================"""
        epoch_loss = train_val_utils.train_one_epoch(dataloader=train_loader, model=model, device=device,
                                                     optimizer=optimizer, loss_function=loss_function,
                                                     lr_scheduler=lr_scheduler, epoch=epoch, logger=logger, args=args)
        val_loss, globle_acc = train_val_utils.evaluate_one_epoch(dataloader=val_loader, model=model,
                                                                  device=device, logger=logger, args=args,
                                                                  epoch=epoch, loss_function=loss_function)

        """按需求保存模型。可以每个epoch都保存一次模型，也可以只保存最优模型"""
        if globle_acc > best_metric:
            best_metric = globle_acc
            torch.save(model.state_dict(), os.path.join(args.model_save_path, f"model_weight{epoch}.pth"))

        # 每个epoch，追加一个画图所需要用的数组的元素
        epochs_plt = np.append(epochs_plt, epoch)
        train_loss_plt = np.append(train_loss_plt, epoch_loss)
        val_loss_plt = np.append(val_loss_plt, val_loss)
        acc_plt = np.append(acc_plt, globle_acc.item())

        # 每隔5个epoch，画一次图
        if (epoch+1) % 5 == 0:
            plt_xOy(x_ndarray=epochs_plt, y_ndarray=[train_loss_plt, val_loss_plt],  args=args, figure_num=0,
                    png_name="Xception_loss.png", x_axis_label="epoch", y_axis_label="loss",
                    legend=["train_loss", "val_loss"])
            plt_xOy(x_ndarray=epochs_plt, y_ndarray=[acc_plt], args=args, png_name="Xception_ACC.png",
                    x_axis_label="epoch", figure_num=1, y_axis_label="100%", legend=["ACC"])

if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path)
    if not os.path.exists("./log"):
        os.makedirs("./log")
    main(args)
