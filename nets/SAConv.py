import torch  # 导入 PyTorch 库
import torch.nn.functional as F  # 导入 PyTorch 的函数库
import torch.nn as nn  # 导入 PyTorch 的神经网络模块



def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
class SAConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, s=1, p=None, g=1, d=1, act=True, bias=True):
        super().__init__(
            in_channels, out_channels, kernel_size, stride=s, padding=autopad(kernel_size, p), dilation=d, groups=g,
            bias=bias)
        # 定义一个1x1卷积作为开关机制
        self.switch = torch.nn.Conv2d(
            self.in_channels, 1, kernel_size=1, stride=s, bias=True)
        self.switch.weight.data.fill_(0)
        self.switch.bias.data.fill_(1)
        # 参数，用于调整权重差异
        self.weight_diff = torch.nn.Parameter(torch.Tensor(self.weight.size()))
        self.weight_diff.data.zero_()
        # 预先和后续的1x1卷积，用于实现上下文依赖
        self.pre_context = torch.nn.Conv2d(
            self.in_channels, self.in_channels, kernel_size=1, bias=True)
        self.pre_context.weight.data.fill_(0)
        self.pre_context.bias.data.fill_(0)
        self.post_context = torch.nn.Conv2d(
            self.out_channels, self.out_channels, kernel_size=1, bias=True)
        self.post_context.weight.data.fill_(0)
        self.post_context.bias.data.fill_(0)

        # 批归一化和激活函数
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())



    def forward(self, x):
        # 前置上下文模块，增强输入特征
        avg_x = torch.nn.functional.adaptive_avg_pool2d(x, output_size=1)
        avg_x = self.pre_context(avg_x)
        avg_x = avg_x.expand_as(x)
        x = x + avg_x
        # 使用开关控制不同的卷积核
        avg_x = torch.nn.functional.pad(x, pad=(2, 2, 2, 2), mode="reflect")
        avg_x = torch.nn.functional.avg_pool2d(avg_x, kernel_size=5, stride=1, padding=0)
        switch = self.switch(avg_x)
        # 标准化权重并进行卷积
        weight = self._get_weight(self.weight)
        out_s = super()._conv_forward(x, weight, None)
        ori_p = self.padding
        ori_d = self.dilation
        self.padding = tuple(3 * p for p in self.padding)
        self.dilation = tuple(3 * d for d in self.dilation)
        weight = weight + self.weight_diff
        out_l = super()._conv_forward(x, weight, None)
        out = switch * out_s + (1 - switch) * out_l
        self.padding = ori_p
        self.dilation = ori_d
        # 后置上下文模块，增强输出特征
        avg_x = torch.nn.functional.adaptive_avg_pool2d(out, output_size=1)
        avg_x = self.post_context(avg_x)
        avg_x = avg_x.expand_as(out)
        out = out + avg_x
        return self.act(self.bn(out))
if __name__ == '__main__':
    x = torch.randn(1, 256, 16, 16)  # 创建随机输入张量
    model = SAConv2d(256 , 256 , 1)  # 创建 ScConv 模型
    print(model(x).shape)  # 打印模型输出的形状
