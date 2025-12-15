import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange


class CAFM(nn.Module):  # Cross Attention Fusion Module
    def __init__(self, channels):
        super(CAFM, self).__init__()

        self.conv1_spatial = nn.Conv2d(2, 1, 3, stride=1, padding=1, groups=1)
        self.conv2_spatial = nn.Conv2d(1, 1, 3, stride=1, padding=1, groups=1)

        self.avg1 = nn.Conv2d(channels, 64, 1, stride=1, padding=0)
        self.avg2 = nn.Conv2d(channels, 64, 1, stride=1, padding=0)
        self.max1 = nn.Conv2d(channels, 64, 1, stride=1, padding=0)
        self.max2 = nn.Conv2d(channels, 64, 1, stride=1, padding=0)

        self.avg11 = nn.Conv2d(64, channels, 1, stride=1, padding=0)
        self.avg22 = nn.Conv2d(64, channels, 1, stride=1, padding=0)
        self.max11 = nn.Conv2d(64, channels, 1, stride=1, padding=0)
        self.max22 = nn.Conv2d(64, channels, 1, stride=1, padding=0)

    def forward(self, f1, f2):
        b, c, h, w = f1.size()

        f1 = f1.reshape([b, c, -1])
        f2 = f2.reshape([b, c, -1])

        avg_1 = torch.mean(f1, dim=-1, keepdim=True).unsqueeze(-1)
        max_1, _ = torch.max(f1, dim=-1, keepdim=True)
        max_1 = max_1.unsqueeze(-1)

        avg_1 = F.relu(self.avg1(avg_1))
        max_1 = F.relu(self.max1(max_1))
        avg_1 = self.avg11(avg_1).squeeze(-1)
        max_1 = self.max11(max_1).squeeze(-1)
        a1 = avg_1 + max_1

        avg_2 = torch.mean(f2, dim=-1, keepdim=True).unsqueeze(-1)
        max_2, _ = torch.max(f2, dim=-1, keepdim=True)
        max_2 = max_2.unsqueeze(-1)

        avg_2 = F.relu(self.avg2(avg_2))
        max_2 = F.relu(self.max2(max_2))
        avg_2 = self.avg22(avg_2).squeeze(-1)
        max_2 = self.max22(max_2).squeeze(-1)
        a2 = avg_2 + max_2

        cross = torch.matmul(a1, a2.transpose(1, 2))

        a1 = torch.matmul(F.softmax(cross, dim=-1), f1)
        a2 = torch.matmul(F.softmax(cross.transpose(1, 2), dim=-1), f2)

        a1 = a1.reshape([b, c, h, w])
        avg_out = torch.mean(a1, dim=1, keepdim=True)
        max_out, _ = torch.max(a1, dim=1, keepdim=True)
        a1 = torch.cat([avg_out, max_out], dim=1)
        a1 = F.relu(self.conv1_spatial(a1))
        a1 = self.conv2_spatial(a1)
        a1 = a1.reshape([b, 1, -1])
        a1 = F.softmax(a1, dim=-1)

        a2 = a2.reshape([b, c, h, w])
        avg_out = torch.mean(a2, dim=1, keepdim=True)
        max_out, _ = torch.max(a2, dim=1, keepdim=True)
        a2 = torch.cat([avg_out, max_out], dim=1)
        a2 = F.relu(self.conv1_spatial(a2))
        a2 = self.conv2_spatial(a2)
        a2 = a2.reshape([b, 1, -1])
        a2 = F.softmax(a2, dim=-1)

        f1 = f1 * a1 + f1
        f2 = f2 * a2 + f2

        f1 = f1.squeeze(0)
        f2 = f2.squeeze(0)

        return f1.transpose(0, 1), f2.transpose(0, 1)




if __name__ == '__main__':
    x = torch.randn(1, 256 , 40 , 40)  # 创建随机输入张量
    model = CAFM(256)  # 创建 ScConv 模型
    print(model(x).shape)  # 打印模型输出的形状