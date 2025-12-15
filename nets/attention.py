from multiprocessing import reduction

import torch
import torch.nn as nn
import math

#SEnet通道注意力机制
# 其具体实现方式就是：
# 1、对输入进来的特征层进行全局平均池化。
# 2、然后进行两次全连接，第一次全连接神经元个数较少，第二次全连接神经元个数和输入特征层相同。
# 3、在完成两次全连接后，我们再取一次Sigmoid将值固定到0 - 1之间，此时我们获得了输入特征层每一个通道的权值（0 - 1之间）。
# 4、在获得这个权值后，我们将这个权值乘上原输入特征层即可。

class se_block(nn.Module):
    def __init__(self, channel, ratio=16):
        super(se_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // ratio, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(channel // ratio, channel, bias=False),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
# 对过程的可视化显示，观察其结构
# model = se_block(512)
# print(model)
# inputs = torch.ones([2,512,26,26])
# outputs = model(inputs)

#CBAM注意力机制
    # CBAM将通道注意力机制和空间注意力机制进行一个结合，相比于SENet只关注通道的注意力机制可以取得更好的效果。
    #CBAM会对输入进来的特征层，分别进行通道注意力机制的处理和空间注意力机制的处理。
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 利用1x1卷积代替全连接
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        # 观察通道注意力机制的结构
        # print(self.sigmoid(out))
        return   self.sigmoid(out)
# model = ChannelAttention(512)
# print(model)
# inputs = torch.ones([2,512,26,26])
# outputs = model(inputs)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        x = self.sigmoid(x)
        # 观察空间注意力机制的结构
        # print(x)
        return self.sigmoid(x)

class cbam_block(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7):
        super(cbam_block, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x
# 对过程的可视化显示，观察其结构
# model = cbam_block(512)
# print(model)
# inputs = torch.ones([2,512,26,26])
# outputs = model(inputs)

#ECANet注意力机制
    # ECANet是也是通道注意力机制的一种实现形式。ECANet可以看作是SENet的改进版。
    # ECANet的作者认为SENet对通道注意力机制的预测带来了副作用，捕获所有通道的依赖关系是低效并且是不必要的。
    # 在ECANet的论文中，作者认为卷积具有良好的跨通道信息获取能力。
    #ECA模块的思想是非常简单的，它去除了原来SE模块中的全连接层，直接在全局平均池化之后的特征上通过一个1D卷积进行学习。
class eca_block(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(eca_block, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        # y都在0-1之间，代表每一个通道的权值
        # print(y)
        return x * y.expand_as(x)
# 对过程的可视化显示，观察其结构
# model = eca_block(512)
# print(model)
# inputs = torch.ones([2,512,26,26])
# outputs = model(inputs)

# CA注意力机制
# 作者认为现有的注意力机制（如CBAM、SE）在求取通道注意力的时候，通道的处理一般是采用全局最大池化/平均池化，这样会损失掉物体的空间信息。
# 作者期望在引入通道注意力机制的同时，引入空间注意力机制，作者提出的注意力机制将位置信息嵌入到了通道注意力中。

class CA_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CA_Block, self).__init__()

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
                                  bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)

        self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        # batch_size, c, h, w
        _, _, h, w = x.size()

        # batch_size, c, h, w => batch_size, c, h, 1 => batch_size, c, 1, h
        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        # batch_size, c, h, w => batch_size, c, 1, w
        x_w = torch.mean(x, dim=2, keepdim=True)

        # batch_size, c, 1, w cat batch_size, c, 1, h => batch_size, c, 1, w + h
        # batch_size, c, 1, w + h => batch_size, c / r, 1, w + h
        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))

        # batch_size, c / r, 1, w + h => batch_size, c / r, 1, h and batch_size, c / r, 1, w
        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)

        # batch_size, c / r, 1, h => batch_size, c / r, h, 1 => batch_size, c, h, 1
        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        # batch_size, c / r, 1, w => batch_size, c, 1, w
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        return out

class CA(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7):
        super(CA, self).__init__()
        self.CA = CA_Block(channel)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = self.CA(x)
        x = x * self.spatialattention(x)
        return x
# 对过程的可视化显示，观察其结构
model = CA(512)
print(model)
inputs = torch.ones([2,512,26,26])
outputs = model(inputs)

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
# model = SimAM(512)
# print(model)
# inputs = torch.ones([2,512,26,26])
# outputs = model(inputs)
# print(outputs.shape)
#===========================================EMA注意力机制============================================================
import torch  # 导入 PyTorch 库
from torch import nn  # 从 PyTorch 中导入神经网络模块


class EMA(nn.Module):  # 定义一个继承自 nn.Module 的 EMA 类
    def __init__(self, channels, c2=None, factor=32):  # 构造函数，初始化对象
        super(EMA, self).__init__()  # 调用父类的构造函数
        self.groups = factor  # 定义组的数量为 factor，默认值为 32
        assert channels // self.groups > 0  # 确保通道数可以被组数整除
        self.softmax = nn.Softmax(-1)  # 定义 softmax 层，用于最后一个维度
        self.agp = nn.AdaptiveAvgPool2d((1, 1))  # 定义自适应平均池化层，输出大小为 1x1
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 定义自适应平均池化层，只在宽度上池化
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # 定义自适应平均池化层，只在高度上池化
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)  # 定义组归一化层
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1,
                                 padding=0)  # 定义 1x1 卷积层
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1,
                                 padding=1)  # 定义 3x3 卷积层

    def forward(self, x):  # 定义前向传播函数
        b, c, h, w = x.size()  # 获取输入张量的大小：批次、通道、高度和宽度
        group_x = x.reshape(b * self.groups, -1, h, w)  # 将输入张量重新形状为 (b * 组数, c // 组数, 高度, 宽度)
        x_h = self.pool_h(group_x)  # 在高度上进行池化
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)  # 在宽度上进行池化并交换维度
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))  # 将池化结果拼接并通过 1x1 卷积层
        x_h, x_w = torch.split(hw, [h, w], dim=2)  # 将卷积结果按高度和宽度分割
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())  # 进行组归一化，并结合高度和宽度的激活结果
        x2 = self.conv3x3(group_x)  # 通过 3x3 卷积层
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))  # 对 x1 进行池化、形状变换、并应用 softmax
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # 将 x2 重新形状为 (b * 组数, c // 组数, 高度 * 宽度)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))  # 对 x2 进行池化、形状变换、并应用 softmax
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # 将 x1 重新形状为 (b * 组数, c // 组数, 高度 * 宽度)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)  # 计算权重
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)  # 应用权重并将形状恢复为原始大小
# model = EMA(512)
# print(model)
# inputs = torch.ones([2,512,26,26])
# outputs = model(inputs)
# print(outputs.shape)
#===========================================Triple Attention===================================================
# import torch
# import torch.nn as nn
#
#
# # 定义一个基础的卷积模块
# class BasicConv(nn.Module):
#     def __init__(
#             self,
#             in_planes,  # 输入通道数
#             out_planes,  # 输出通道数
#             kernel_size,  # 卷积核大小
#             stride=1,  # 步长
#             padding=0,  # 填充
#             dilation=1,  # 空洞率
#             groups=1,  # 分组卷积的组数
#             relu=True,  # 是否使用ReLU激活函数
#             bn=True,  # 是否使用批标准化
#             bias=False,  # 卷积是否添加偏置
#     ):
#         super(BasicConv, self).__init__()
#         self.out_channels = out_planes
#         # 定义卷积层
#         self.conv = nn.Conv2d(
#             in_planes,
#             out_planes,
#             kernel_size=kernel_size,
#             stride=stride,
#             padding=padding,
#             dilation=dilation,
#             groups=groups,
#             bias=bias,
#         )
#         # 可选的批标准化层
#         self.bn = (
#             nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
#             if bn
#             else None
#         )
#         # 可选的ReLU激活层
#         self.relu = nn.ReLU() if relu else None
#
#     def forward(self, x):
#         x = self.conv(x)
#         if self.bn is not None:
#             x = self.bn(x)
#         if self.relu is not None:
#             x = self.relu(x)
#         return x
#
#
# # 定义一个通道池化模块
# class ChannelPool(nn.Module):
#     def forward(self, x):
#         return torch.cat(
#             (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
#         )
#
#
# # 定义一个空间门控模块
# class SpatialGate(nn.Module):
#     def __init__(self):
#         super(SpatialGate, self).__init__()
#         kernel_size = 7
#         self.compress = ChannelPool()
#         self.spatial = BasicConv(
#             2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False
#         )
#
#     def forward(self, x):
#         x_compress = self.compress(x)
#         x_out = self.spatial(x_compress)
#         scale = torch.sigmoid_(x_out)
#         return x * scale
#
#
# # 定义一个三元注意力模块
# class TripletAttention(nn.Module):
#     def __init__(
#             self,
#             gate_channels,  # 门控通道数
#             reduction_ratio=16,  # 缩减比率
#             pool_types=["avg", "max"],  # 池化类型
#             no_spatial=False,  # 是否禁用空间门控
#     ):
#         super(TripletAttention, self).__init__()
#         self.ChannelGateH = SpatialGate()
#         self.ChannelGateW = SpatialGate()
#         self.no_spatial = no_spatial
#         if not no_spatial:
#             self.SpatialGate = SpatialGate()
#
#     def forward(self, x):
#         x_perm1 = x.permute(0, 2, 1, 3).contiguous()
#         x_out1 = self.ChannelGateH(x_perm1)
#         x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
#         x_perm2 = x.permute(0, 3, 2, 1).contiguous()
#         x_out2 = self.ChannelGateW(x_perm2)
#         x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
#         if not self.no_spatial:
#             x_out = self.SpatialGate(x)
#             x_out = (1 / 3) * (x_out + x_out11 + x_out21)
#         else:
#             x_out = (1 / 2) * (x_out11 + x_out21)
#         return x_out
# model = TripletAttention(512)
# print(model)
# inputs = torch.ones([2,512,26,26])
# outputs = model(inputs)
# print(outputs.shape)