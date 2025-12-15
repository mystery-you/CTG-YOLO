import torch
import torch.nn as nn


def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1,
                 act=nn.LeakyReLU(0.1, inplace=True)):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = nn.LeakyReLU(0.1, inplace=True) if act is True else (
            act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))



class GSConv(nn.Module):
    # GSConv https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, None, g, act)
        #self.cv3 = ScConv(c_)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)
        #self.cv2 = WTConv2d(c_, c_)
        #self.cv2 = PConv(c_ , c_)
        #self.cv2 = ScConv1(c_)

    def forward(self, x):
        x1 = self.cv1(x)
        #x1 = self.cv3(x1)
        x2 = torch.cat((x1, self.cv2(x1) ), 1)
        # shuffle
        y = x2.reshape(x2.shape[0], 2, x2.shape[1] // 2, x2.shape[2], x2.shape[3])
        y = y.permute(0, 2, 1, 3, 4)
        return y.reshape(y.shape[0], -1, y.shape[3], y.shape[4])
# if __name__ == '__main__':
#     x = torch.randn(1, 256, 16, 16)  # 创建随机输入张量
#     model = GSConv(256 , 128)  # 创建 ScConv 模型
#     print(model(x).shape)  # 打印模型输出的形状

class GSBottleneck(nn.Module):
    # GS Bottleneck https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        c_ = c2 // 2
        # for lighting
        self.conv_lighting = nn.Sequential(
            GSConv(c1, c_, 1, 1),
            GSConv(c_, c2, 1, 1, act=False))
        # for receptive field
        self.conv = nn.Sequential(  # 没用到
            GSConv(c1, c_, 3, 1),
            GSConv(c_, c2, 3, 1, act=False))
        self.shortcut = Conv(c1, c2, 3, 1, act=False)
        #self.shortcut1 = ScConv(c2)

    def forward(self, x):
        return self.conv_lighting(x) + self.shortcut(x)
        #return self.conv_lighting(x) + self.shortcut1(self.shortcut(x))
if __name__ == '__main__':
    x = torch.randn(1, 128, 16, 16)  # 创建随机输入张量
    model = GSBottleneck(128 , 128)  # 创建 ScConv 模型
    print(model(x).shape)  # 打印模型输出的形状


class VoVGSCSP(nn.Module):
    # VoV-GSCSP https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(GSBottleneck(c_, c_) for _ in range(n)))

    def forward(self, x):
        x1 = self.cv1(x)
        return self.cv2(torch.cat((self.m(x1), x1), dim=1))

if __name__ == '__main__':
    x = torch.randn(1, 256 , 40 , 40)  # 创建随机输入张量
    model = VoVGSCSP(256 , 128)  # 创建 ScConv 模型
    print(model(x).shape)  # 打印模型输出的形状
