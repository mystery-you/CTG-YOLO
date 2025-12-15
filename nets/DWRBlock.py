import torch
import torch.nn as nn


class DWRBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DWRBlock, self).__init__()
        # 分支 1: 1x1 卷积
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # 分支 2: 3x3 卷积
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # 分支 3: 5x5 卷积
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # 特征融合卷积
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # 残差连接
        self.residual_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        # 多分支卷积
        branch1_out = self.branch1(x)
        branch2_out = self.branch2(x)
        branch3_out = self.branch3(x)
        # 特征融合
        fused_feat = torch.cat([branch1_out, branch2_out, branch3_out], dim=1)
        fused_feat = self.fusion_conv(fused_feat)
        # 残差连接
        residual = self.residual_conv(x)
        output = fused_feat + residual
        return output