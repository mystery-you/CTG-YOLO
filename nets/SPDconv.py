# SPD-Conv
import torch
import torch.nn as nn

def autopad(k, p=None, d=1):
    # kernel, padding, dilation
    # 对输入的特征层进行自动padding，按照Same原则
    if d > 1:
        # actual kernel-size
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        # auto-pad
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class SiLU(nn.Module):
    # SiLU激活函数
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


class Conv(nn.Module):
    # 标准卷积+标准化+激活函数
    default_act = SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class Spdconv(nn.Module):
    # def __init__(self,c1,c2,dimension=1):
    def __init__(self, c1, c2):
        super().__init__()
        # self.d = dimension
        c1 = c1 * 4
        self.conv = Conv(c1,c2)

    def forward(self,x):
        # return torch.cat([x[...,::2,::2],x[...,1::2,::2],x[...,::2,1::2],x[...,1::2,1::2]],1)
        x = torch.cat([x[...,::2,::2],x[...,1::2,::2],x[...,::2,1::2],x[...,1::2,1::2]],1)
        x = self.conv(x)
        return x
if __name__ == '__main__':
    x = torch.randn(1, 256, 16, 16)  # 创建随机输入张量
    model = Spdconv(256 , 256)  # 创建 ScConv 模型
    print(model(x).shape)  # 打印模型输出的形状