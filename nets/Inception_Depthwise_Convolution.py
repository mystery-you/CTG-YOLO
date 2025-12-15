# 针对传统的大 Kernel Depthwise Convolution 阻碍模型速度的问题，本文提出了 Inception Depthwise Convolution。
# Inception 这个模型利用了小 Kernel (如 3×3) 和大 Kernel (如 5×5) 的几个分支。
# 同样地，Inception Depthwise Convolution 采用了 3×3 作为基本分支之一，但避免了大的矩形 Kernel，因为它们的实际速度较慢。
# 大的矩形 Kernel 被分解为 1×Kw和 Kh×1.
import torch.nn as nn
import torch


class InceptionDWConv2d(nn.Module):
    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=1 / 8):
        super().__init__()

        gc = int(in_channels * branch_ratio)  # channel number of a convolution branch

        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)

        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2),
                                  groups=gc)

        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),
                                  groups=gc)

        self.split_indexes = (gc, gc, gc, in_channels - 3 * gc)

    def forward(self, x):
        # B, C, H, W = x.shape
        x_hw, x_w, x_h, x_id = torch.split(x, self.split_indexes, dim=1)

        return torch.cat(
            (self.dwconv_hw(x_hw),
             self.dwconv_w(x_w),
             self.dwconv_h(x_h),
             x_id),
            dim=1)
if __name__ == '__main__':
    x = torch.randn(1, 128, 16, 16)  # 创建随机输入张量
    model = InceptionDWConv2d(128)  # 创建InceptionDWConv2d模型
    print(model(x).shape)  # 打印模型输出的形状