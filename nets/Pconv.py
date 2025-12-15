import torch
import torch.nn as nn

class PConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, n_div=4):
        super(PConv, self).__init__()
        self.partial_channels = in_channels // n_div
        self.remaining_channels = in_channels - self.partial_channels
        self.partial_conv = nn.Conv2d(self.partial_channels, self.partial_channels, kernel_size, stride, padding, bias=bias)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=bias)

    def forward(self, x):
        x_partial = self.partial_conv(x[:, :self.partial_channels, :, :])
        x_remaining = x[:, self.partial_channels:, :, :]
        x = torch.cat((x_partial, x_remaining), 1)
        x = self.pointwise_conv(x)
        return x
if __name__ == '__main__':
    x = torch.randn(1, 256, 16, 16)  # 创建随机输入张量
    model = PConv(256,512)  # 创建 ScConv 模型
    print(model(x).shape)  # 打印模型输出的形状