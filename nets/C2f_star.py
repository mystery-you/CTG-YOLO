import torch
import torch.nn as nn
from .ConVNext_Block import ConvNextBlock
from .GSconv import GSConv , GSBottleneck
from .starBlocks import StarNetBlock
from .Inception_Depthwise_Convolution import InceptionDWConv2d
from .ShuffleAttention import ShuffleAttention

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
        self.conv   = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.act    = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class Bottleneck(nn.Module):
    # 标准瓶颈结构，残差结构
    # c1为输入通道数，c2为输出通道数
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        # self.SA = ShuffleAttention(c_)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        # self.cv1 = GSConv(c1, c_)
        # self.cv2 = GSConv(c_, c2, g = g)
        # self.cv1 = DeformConv2d(c1, c_)
        # self.cv2 = DeformConv2d(c_, c2)
        self.add = shortcut and c1 == c2


    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
        # return x + self.cv2(self.SA(self.cv1(x))) if self.add else self.cv2(self.SA(self.cv1(x)))
class C2f_star(nn.Module):
    # CSPNet结构结构，大残差结构
    # c1为输入通道数，c2为输出通道数
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c      = int(c2 * e)
        self.cv1    = Conv(c1, 2 * self.c, 1, 1)
        self.cv2    = Conv((2 + n) * self.c, c2, 1)
        # self.m      = nn.ModuleList(StarNetBlock(self.c) for _ in range(n))
        # self.m      = nn.ModuleList(ConvNextBlock(self.c, shortcut, g) for _ in range(n))
        # self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g) for _ in range(n))
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        # self.m = nn.ModuleList(GSBottleneck(self.c, self.c) for _ in range(n))
        self.cv3 = ConvNextBlock(c1)
        # self.cv3 = InceptionDWConv2d(c1)
        # self.cv4 = StarNetBlock(c1)
        # self.cbam = cbam_block(c1)
        # self.ca = CA_Block(512)
        # self.SiMam = SimAM()
        self.cv5 = Conv(2 * c1 , c2)
    def forward(self, x):
        # 进行一个卷积，然后划分成两份，每个通道都为c
        y = list(self.cv1(x).split((self.c, self.c), 1))
        # 每进行一次残差结构都保留，然后堆叠在一起，密集残差
        y.extend(m(y[-1]) for m in self.m)
        # return self.cv2(torch.cat(y, 1))
        y = self.cv2(torch.cat(y, 1))
        # return self.cv4(torch.cat((self.SiMam(y),self.cv3(x)), 1))
        return self.cv5(torch.cat((y, self.cv3(x)), 1))
        # return self.cv5(torch.cat((self.cv4(x), self.cv3(x)), 1))
        # x2 = torch.cat((self.SiMam(y),self.cv3(x)), 1)
        # # shuffle
        # y = x2.reshape(x2.shape[0], 2, x2.shape[1] // 2, x2.shape[2], x2.shape[3])
        # y = y.permute(0, 2, 1, 3, 4)
        # return self.cv4(y.reshape(y.shape[0], -1, y.shape[3], y.shape[4]))