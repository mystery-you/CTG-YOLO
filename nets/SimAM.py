import torch
import torch.nn as nn


class SimAM(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(SimAM, self).__init__()

        # 初始化Sigmoid激活函数和e_lambda参数
        self.activation = nn.Sigmoid()  # Sigmoid激活函数用于映射输出到(0, 1)之间
        self.e_lambda = e_lambda  # 控制分母的平滑参数

    def __repr__(self):
        # 返回模型的字符串表示，包括e_lambda参数的值
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        # 静态方法，返回模型的名称
        return "simam"

    def forward(self, x):
        # 前向传播函数，接收输入张量x，返回处理后的张量

        b, c, h, w = x.size()  # 获取输入张量的batch大小、通道数、高度和宽度

        n = w * h - 1  # 计算像素数量减一，用于标准化

        # 计算每个像素与平均值的差的平方
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)

        # 计算SimAM激活函数的输出
        # 分子部分：每个像素的平方差除以分母的加权平均
        # 加上0.5是为了映射输出到(0.5, 1)之间
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        # 返回经过SimAM激活函数处理后的特征图
        return x * self.activation(y)