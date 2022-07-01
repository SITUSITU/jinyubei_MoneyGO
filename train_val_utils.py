import torch.nn as nn
import torch
import os

"""自定义损失函数的方法"""
class XXX_loss(nn.Module):
    def __init__(self, ):
        super(XXX_loss, self).__init__()
        """在这里用self获取所需要的超参数"""

    def forward(self, predict, label, xx):
        """在这里利用init传进来的参数和forward的参数进行损失的计算"""
        loss = (predict - label)**2
        return loss


def train_one_epoch(dataloader, model, device, optimizer, loss_function, lr_scheduler, epoch, logger, args):
    model.train()
    epoch_loss = 0.0
    for step, data in enumerate(dataloader):  # batch循环，每个循环会把一个batch的数据送入网络
        batched_inputs, batched_labels = data
        batched_inputs, batched_labels = batched_inputs.to(device), batched_labels.to(device)

        optimizer.zero_grad()  # 梯度清零

        batched_outputs = model(batched_inputs.to(device))  # 前向传播获得输出

        batched_loss = loss_function(batched_outputs, batched_labels.to(device))

        batched_loss.backward()  # 反向传播获得梯度

        optimizer.step()  # 根据梯度进行更新

        epoch_loss += batched_loss.item()  # 把每个 batch 的 loss 都加起来，最终返回整个 epoch 的 loss

        print(f"training------epoch:[{epoch}/{args.epochs}]---steps:[{step + 1}/{len(dataloader)}]---batched_loss: {batched_loss}---lr={optimizer.param_groups[0]['lr']} ")

    lr_scheduler.step()
    logger.info(f"training------epoch:[{epoch}/{args.epochs}]---epoch_loss: {epoch_loss}---lr={optimizer.param_groups[0]['lr']}")
    return epoch_loss


def evaluate_one_epoch(dataloader, model, device, logger, args, epoch, loss_function):
    """
    送入验证一个 epoch 所需要的元素，返回自定义的指标

    :param dataloader:
    :param model:
    :param device:
    :param logger:
    :param args:
    :param epoch:
    :param loss_function:
    :return:
    """

    mat = Confusion_Matrix(num_classes=args.num_classes)
    mat.reset()
    model.eval()
    val_epoch_loss = 0.0

    with torch.no_grad():  # 关闭梯度记录
        for step, data in enumerate(dataloader):
            batched_inputs, batched_labels = data
            batched_inputs, batched_labels = batched_inputs.to(device), batched_labels.to(device)

            batched_outputs = model(batched_inputs)
            batched_loss = loss_function(batched_outputs, batched_labels)
            val_epoch_loss += batched_loss.item()

            """在这里利用验证集的output获得predict生成混淆矩阵，计算各种指标"""
            batched_predict = batched_outputs.argmax(1)  # 获取channel维度最大值索引
            mat.update(label=batched_labels, predict=batched_predict)
            print(f"validating------step:[{step + 1}/{len(dataloader)}]")

    """在这里可以计算当前epoch的各项指标，并打印保存结果result"""
    Recall = mat.compute_Recall_or_specificity_with_sensitivity()
    Precision = mat.compute_precision()
    globle_acc = mat.compute_globle_acc()

    logger.info(f"validating------epoch[{epoch}/{args.epochs}]---Recall[{Recall.cpu().numpy()}---Precision[{Precision.cpu().numpy()}]---acc[{globle_acc}]")

    return val_epoch_loss, globle_acc


class Confusion_Matrix(object):
    """
    传入分类个数 num_classes ，创建一个 num_classes×num_classes的空矩阵
    """
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None  # 构建空混淆矩阵

    def update(self, label, predict):
        """
        这里送进来的必须是一维的tensor，对于分类网络，则直接送进来
        对于分割网络，则可以展平后送进来，对统计结果不产生影响
        :param target: 一维tensor
        :param predict: 一维tensor
        :return: 根据送进来的tensor，统计并更新混淆矩阵的值，其中纵坐标是label，横坐标是predict
        """
        n = self.num_classes  # 获取分类个数n（包括背景）
        if self.mat is None:
            # 创建n行n列的混淆矩阵
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=label.device)
        with torch.no_grad():
            k = (label >= 0) & (label < n)  # 寻找真实值中为目标的像素索引。一般类别数都包含了标签，且难以判别的都赋值255
            # 所以这行代码既能找到非背景类别所在位置，又能忽略255像素值
            # 这里返回的k的形式如下：tensor([ True,  True,  True,  True,  True, False,  True,  True, False, False, True])

            inds = n * label[k].to(torch.int64) + predict[k]  # 统计像素真实类别a[k]被预测成类别b[k]的个数(这里的做法很巧妙)
            # 将标签中所有非背景类别的数值乘 类别数n。然后加上预测值的类别数值
            # 能得到新的一维tensor，该tensor元素的值即在混淆矩阵中应该在哪个位置 +1

            self.mat += torch.bincount(inds.int(), minlength=n ** 2).reshape(n, n)
            # 用统计直方图计算新tensor各个数值出现的次数，并将统计结果reshape回n*n。即可获得混淆矩阵
            # 其中，纵坐标为真实值，横坐标为预测值

    def compute_globle_acc(self):
        """
        根据update函数更新的混淆矩阵，计算全局准确率
        :return: 0维tensor
        """
        mat = self.mat.float()  # 把混淆矩阵转换为float类型
        globle_acc = torch.diag(mat).sum() / mat.sum()

        return globle_acc

    def compute_IoU(self):
        """
        根据update函数更新的混淆矩阵，计算每个分类的IoU
        :return: 一维tensor，共num_class个元素，每个元素代表对应类的IoU
        """
        mat = self.mat.float()
        IoU = torch.diag(mat) / (mat.sum(1) + mat.sum(0) - torch.diag(mat))

        return IoU

    def compute_Recall_or_specificity_with_sensitivity(self):
        """
        根据update函数更新的混淆矩阵，计算每个分类中，预测正确的比例
        :return: 一维tensor，共num_class个元素
        如果是多分类，即num_class>2
            则tensor每个元素代表了每个分类的召回率（找准率）
        如果是二分类，即num_class=2
            则tensor第一个元素代表了特异性，因为第一类0是负样本。负样本中预测正确的个数即为特异性
            第二个元素代表了敏感度，因为第二类1是正样本，正样本中预测正确的个数即为敏感度
        """
        mat = self.mat.float()
        Recall_or_specificity_with_sensitivity = torch.diag(mat) / mat.sum(1)
        return Recall_or_specificity_with_sensitivity

    def compute_precision(self):
        """
        根据update函数更新的混淆矩阵，计算预测为x类的结果中，预测正确的比例
        :return: 一维tensor，共num_class个元素，每个元素代表对应预测为该种类的正确率
        """
        mat = self.mat.float()
        precision = torch.diag(mat) / mat.sum(0)
        return precision

    def compute_F1_score(self):
        """
        利用召回率R和精确率P，综合计算F1
        F1 = 2*P*R/（P+R）
        :return: 一维tensor，每个元素为对应类的F1指标
        """
        R = self.compute_Recall_or_specificity_with_sensitivity()
        P = self.compute_precision()
        F1 = 2 * P * R /(P + R)
        return F1

    def reset(self):
        if self.mat is not None:
            self.mat.zero_()


if __name__ == "__main__":
    a = torch.tensor([[0, 1, 1], [2, 2, 3], [1, 2, 3]])
    b = torch.tensor([[1, 1, 1], [2, 3, 3], [1, 2, 3]])

    mat = Confusion_Matrix(num_classes=4)
    mat.update(a.flatten(), b.flatten())
    acc = mat.compute_globle_acc()
    recall = mat.compute_Recall_or_specificity_with_sensitivity()
    print(acc)