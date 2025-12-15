import torch
import torch.nn as nn

#========================================DropPath=====================
#DropPath与Dropout的主要区别在于它们应用的层次不同。Dropout随机丢弃单个神经元的输出，而DropPath随机丢弃整个层之间的连接。
# 这使得DropPath在处理具有多分支结构的网络时特别有用，如ResNet或Transformer模型。
#通过这种方式，DropPath有助于提高模型在不同架构和数据集上的泛化能力，同时也增加了训练的难度。
# 因此，在实际应用中，选择合适的drop_prob值非常重要，以确保模型能够有效地学习并收敛。
# class DropPath(nn.Module):
#     def __init__(self, drop_prob=None):
#         super(DropPath, self).__init__()
#         self.drop_prob = drop_prob
#
#     def forward(self, x):
#         if self.drop_prob == 0. or not self.training:
#             return x
#             keep_prob = 1 - self.drop_prob
#             shape = (x.shape[0],) + (1,) * (x.ndim - 1)
#             random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
#             random_tensor.floor_() # binarize
#             output = x.div(keep_prob) * random_tensor
#             return output
def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:  # drop_prob废弃率=0，或者不是训练的时候，就保持原来不变
        return x
    keep_prob = 1 - drop_prob  # 保持率
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # (b, 1, 1, 1) 元组  ndim 表示几维，图像为4维
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)  # 0-1之间的均匀分布[2,1,1,1]
    random_tensor.floor_()  # 下取整从而确定保存哪些样本 总共有batch个数
    output = x.div(keep_prob) * random_tensor  # 除以 keep_prob 是为了让训练和测试时的期望保持一致
    # 如果keep，则特征值除以 keep_prob；如果drop，则特征值为0
    return output  # 与x的shape保持不变


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


#================================================================================

class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)


class StarNetBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0., g = 1):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, 3, 1, (3 - 1) // 2, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 3, 1, (3 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        # x = input +x
        return x
#==================================================
if __name__ == '__main__':
    x = torch.randn(1, 16, 16, 16)  # 创建随机输入张量
    model = StarNetBlock(16)  # 创建 ScConv 模型
    print(model(x).shape)  # 打印模型输出的形状