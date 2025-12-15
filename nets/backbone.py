import torch
import torch.nn as nn
from .ADown import ADown
from .starBlocks import StarNetBlock
from .GSconv import GSBottleneck , VoVGSCSP
from .SimAM import SimAM
from .DCNConvV2 import DeformConv2d
from .attention import cbam_block,CA_Block
from .ConVNext_Block import ConvNextBlock
from .LDConv import LDConv
from .ELAN import Multi_Concat_Block
from .Triple_C2f import Triple_C2f
from .C3 import C3
from .SGDown import SGDown
from .Res2block import C2f_Res2Block
from .SCconv_C2f import C2fScconv
from .C2f_star import C2f_star
from .swim_transformer import SwinStage
from .ELA_attention import EfficientLocalizationAttention
from .attention import CA

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
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        # self.cv1 = DeformConv2d(c1, c_)
        # self.cv2 = DeformConv2d(c_, c2)
        self.add = shortcut and c1 == c2


    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f(nn.Module):
    # CSPNet结构结构，大残差结构
    # c1为输入通道数，c2为输出通道数
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c      = int(c2 * e) 
        self.cv1    = Conv(c1, 2 * self.c, 1, 1)
        self.cv2    = Conv((2 + n) * self.c, c2, 1)
        self.m      = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        # self.cv3 = VoVGSCSP(c1, c2)
        # self.cv4 = Conv(2 * c2, c2, 1, 1)

    def forward(self, x):
        # 进行一个卷积，然后划分成两份，每个通道都为c
        y = list(self.cv1(x).split((self.c, self.c), 1))
        # 每进行一次残差结构都保留，然后堆叠在一起，密集残差
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
        # y = self.cv2(torch.cat(y, 1))
        # return self.cv4(torch.cat((self.cv3(x), y), dim=1))
#==========================================================
class BottleneckCA(nn.Module):
    # 标准瓶颈结构，残差结构
    # c1为输入通道数，c2为输出通道数
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.CA = CA_Block(c2)
        # self.cv1 = DeformConv2d(c1, c_)
        # self.cv2 = DeformConv2d(c_, c2)
        self.add = shortcut and c1 == c2


    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.CA(self.cv2(self.cv1(x)))
class C2fCA(nn.Module):
    # CSPNet结构结构，大残差结构
    # c1为输入通道数，c2为输出通道数
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c      = int(c2 * e)
        self.cv1    = Conv(c1, 2 * self.c, 1, 1)
        self.cv2    = Conv((2 + n) * self.c, c2, 1)
        self.m      = nn.ModuleList(BottleneckCA(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        # self.cv3 = VoVGSCSP(c1, c2)
        # self.cv4 = Conv(2 * c2, c2, 1, 1)

    def forward(self, x):
        # 进行一个卷积，然后划分成两份，每个通道都为c
        y = list(self.cv1(x).split((self.c, self.c), 1))
        # 每进行一次残差结构都保留，然后堆叠在一起，密集残差
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
#==========================================================
class DeformConv2d1(nn.Module):
    # 标准瓶颈结构，残差结构
    # c1为输入通道数，c2为输出通道数
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        # self.cv1 = DeformConv2d(c1, c_)
        self.cv2 = DeformConv2d(c_, c2)
        # self.cv1 = LDConv(c1, c_)
        # self.cv2 = LDConv(c_, c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2fDeformConv2d1(nn.Module):
    # CSPNet结构结构，大残差结构
    # c1为输入通道数，c2为输出通道数
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c      = int(c2 * e)
        self.cv1    = Conv(c1, 2 * self.c, 1, 1)
        self.cv2    = Conv((2 + n) * self.c, c2, 1)
        self.m      = nn.ModuleList(DeformConv2d1(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        # 进行一个卷积，然后划分成两份，每个通道都为c
        y = list(self.cv1(x).split((self.c, self.c), 1))
        # 每进行一次残差结构都保留，然后堆叠在一起，密集残差
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

#==========================================================
class DWRBlock(nn.Module):
    def __init__(self, in_channels, out_channels , g=1):
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
#==========================================================
class C2fDWR(nn.Module):
    # CSPNet结构结构，大残差结构
    # c1为输入通道数，c2为输出通道数
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c      = int(c2 * e)
        self.cv1    = Conv(c1, 2 * self.c, 1, 1)
        self.cv2    = Conv((2 + n) * self.c, c2, 1)
        self.m      = nn.ModuleList(DWRBlock(self.c, self.c , g ) for _ in range(n))

    def forward(self, x):
        # 进行一个卷积，然后划分成两份，每个通道都为c
        y = list(self.cv1(x).split((self.c, self.c), 1))
        # 每进行一次残差结构都保留，然后堆叠在一起，密集残差
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
#==========================================================
class C2fstar(nn.Module):
    # CSPNet结构结构，大残差结构
    # c1为输入通道数，c2为输出通道数
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c      = int(c2 * e)
        self.cv1    = Conv(c1, 2 * self.c, 1, 1)
        self.cv2    = Conv((2 + n) * self.c, c2, 1)
        # self.m      = nn.ModuleList(StarNetBlock(self.c) for _ in range(n))
        # self.m      = nn.ModuleList(ConvNextBlock(self.c, shortcut, g) for _ in range(n))
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g) for _ in range(n))#扩展因子e=0.5,
        # self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.cv3 = ConvNextBlock(c1)
        # self.cv3 = StarNetBlock(c1)
        # self.cbam = cbam_block(c1)
        # self.ca = CA_Block(512)
        self.SiMam = SimAM()
        self.cv4 = Conv(2 * c1 , c2)
    def forward(self, x):
        # 进行一个卷积，然后划分成两份，每个通道都为c
        y = list(self.cv1(x).split((self.c, self.c), 1))
        # 每进行一次残差结构都保留，然后堆叠在一起，密集残差
        y.extend(m(y[-1]) for m in self.m)
        # return self.cv2(torch.cat(y, 1))
        y = self.cv2(torch.cat(y, 1))
        return self.cv4(torch.cat((self.SiMam(y),self.cv3(x)), 1))
        # return self.cv4(torch.cat((y, self.cv3(x)), 1))
        # x2 = torch.cat((self.SiMam(y),self.cv3(x)), 1)
        # # shuffle
        # y = x2.reshape(x2.shape[0], 2, x2.shape[1] // 2, x2.shape[2], x2.shape[3])
        # y = y.permute(0, 2, 1, 3, 4)
        # return self.cv4(y.reshape(y.shape[0], -1, y.shape[3], y.shape[4]))

#==========================================================
class C2fGSBottleneck(nn.Module):
    # CSPNet结构结构，大残差结构
    # c1为输入通道数，c2为输出通道数
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c      = int(c2 * e)
        self.cv1    = Conv(c1, 2 * self.c, 1, 1)
        self.cv2    = Conv((2 + n) * self.c, c2, 1)
        self.m      = nn.ModuleList(GSBottleneck(self.c, self.c ,shortcut, g) for _ in range(n))
        # self.cv3    = VoVGSCSP(c1 , c2)
        # self.cv4    = Conv(2 * c2 , c2, 1,1)
        # self.SimAM = SimAM()

    def forward(self, x):
        # 进行一个卷积，然后划分成两份，每个通道都为c
        # x = self.SimAM(x)
        y = list(self.cv1(x).split((self.c, self.c), 1))
        # y = list(self.SimAM(self.cv1(x)).split((self.c, self.c), 1))
        # 每进行一次残差结构都保留，然后堆叠在一起，密集残差
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
        # y = self.cv2(torch.cat(y, 1))
        # return self.cv4(torch.cat((self.cv3(x), y), dim=1))
#==========================================================
    
class SPPF(nn.Module):
    # SPP结构，5、9、13最大池化核的最大池化。
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_          = c1 // 2
        self.cv1    = Conv(c1, c_, 1, 1)
        self.cv2    = Conv(c_ * 4, c2, 1, 1)
        self.m      = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        #self.n      = nn.AvgPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        #y3 = self.n(x)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))
        #return self.cv2(torch.cat((x, y1, y2, self.m(y2), y3), 1))

class Backbone(nn.Module):
    def __init__(self, base_channels, base_depth, deep_mul, phi, pretrained=False):
        super().__init__()
        #-----------------------------------------------#
        #   输入图片是3, 640, 640
        #-----------------------------------------------#
        # 3, 640, 640 => 32, 640, 640 => 64, 320, 320
        self.stem = Conv(3, base_channels, 3, 2)
        
        # 64, 320, 320 => 128, 160, 160 => 128, 160, 160
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2),
            C2f(base_channels * 2, base_channels * 2, base_depth, True),
        )
        # 128, 160, 160 => 256, 80, 80 => 256, 80, 80
        self.dark3 = nn.Sequential(
            # SimAM(base_channels * 2),
            # ADown(base_channels * 2, base_channels * 4),
            Conv(base_channels * 2, base_channels * 4, 3, 2),
            # SGDown(base_channels * 2, base_channels * 4),
            C2f(base_channels * 4, base_channels * 4, base_depth * 2, True),
        )
        # 256, 80, 80 => 512, 40, 40 => 512, 40, 40
        self.dark4 = nn.Sequential(
            # SimAM(base_channels * 4),
            # ADown(base_channels * 4, base_channels * 8),
            Conv(base_channels * 4, base_channels * 8, 3, 2),
            # SGDown(base_channels * 4 , base_channels * 8),
            C2f(base_channels * 8, base_channels * 8, base_depth * 2, True),
            # C2fGSBottleneck(base_channels * 8, base_channels * 8, base_depth * 2, True),
            # SwinStage(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), 1, 8, 8),
            # Multi_Concat_Block(base_channels * 8, base_channels * 8),
            # C2f_Res2Block(base_channels * 8, base_channels * 8, base_depth * 2, True),
            # C3(base_channels * 8, base_channels * 8, shortcut=True),
            # Triple_C2f(base_channels * 8, base_channels * 8),
        )
#===============
        # self.dark4 = nn.Sequential(
        #     Conv(base_channels * 4, int(base_channels * 8 * deep_mul), 3, 2),
        #     C2f(int(base_channels * 8 * deep_mul), int(base_channels * 8 * deep_mul), base_depth * 2, True),
        #     SPPF(int(base_channels * 8 * deep_mul), int(base_channels * 8 * deep_mul), k=5)
        # )
#===============
        # 512, 40, 40 => 1024 * deep_mul, 20, 20 => 1024 * deep_mul, 20, 20
        self.dark5 = nn.Sequential(
            # SimAM(base_channels * 8),
            # ADown(base_channels * 8, int(base_channels * 16 * deep_mul)),
            Conv(base_channels * 8, int(base_channels * 16 * deep_mul), 3, 2),
            # SGDown(base_channels * 8,int(base_channels * 16 * deep_mul)),
            C2fstar(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), base_depth, True),
            # C2fGSBottleneck(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), base_depth, True),
            # SwinStage(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul),1,8,8),
            # Multi_Concat_Block(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul)),
            # C2f_Res2Block(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), base_depth, True),
            # C3(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), shortcut=True),
            SPPF(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), k=5),
            # CA(int(base_channels * 16 * deep_mul))
        )
        
        if pretrained:
            url = {
                "n" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_n_backbone_weights.pth',
                "s" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_s_backbone_weights.pth',
                "m" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_m_backbone_weights.pth',
                "l" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_l_backbone_weights.pth',
                "x" : 'https://github.com/bubbliiiing/yolov8-pytorch/releases/download/v1.0/yolov8_x_backbone_weights.pth',
            }[phi]
            checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", model_dir="./model_data")
            self.load_state_dict(checkpoint, strict=False)
            print("Load weights from " + url.split('/')[-1])

    def forward(self, x):
        x = self.stem(x)
        x = self.dark2(x)
        # feat1 = x
        #-----------------------------------------------#
        #   dark3的输出为256, 80, 80，是一个有效特征层
        #-----------------------------------------------#
        x = self.dark3(x)
        feat1 = x
        #-----------------------------------------------#
        #   dark4的输出为512, 40, 40，是一个有效特征层
        #-----------------------------------------------#
        x = self.dark4(x)
        feat2 = x
        #-----------------------------------------------#
        #   dark5的输出为1024 * deep_mul, 20, 20，是一个有效特征层
        #-----------------------------------------------#
        x = self.dark5(x)
        feat3 = x
        return feat1, feat2, feat3
