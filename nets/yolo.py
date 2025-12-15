import numpy as np
import torch
import torch.nn as nn

from .BiFPN import BiFPN_Concat
from .backbone import Backbone, C2f, Conv,C2fCA
from .yolo_training import weights_init
from utils.utils_bbox import make_anchors
from .backbone import SPPF
from .VOV_GSCSPC import VoVGSCSP
from .ASFF import ASFF
from .FasterBlock import C2f_FasterBlock
from .Pconv import PConv
from .GSconv import GSConv
from .EMA_Attention import EMA
from .attention import cbam_block ,EMA,CA_Block , CA
from .DWconv import DWConv
from .Biform_attention import BiLevelRoutingAttention
from .ELA_attention import EfficientLocalizationAttention
from .ADown import ADown

def fuse_conv_and_bn(conv, bn):
    # 混合Conv2d + BatchNorm2d 减少计算量
    # Fuse Conv2d() and BatchNorm2d() layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          dilation=conv.dilation,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    # 准备kernel
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # 准备bias
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv

class DFL(nn.Module):
    # DFL模块
    # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    def __init__(self, c1=16):
        super().__init__()
        self.conv   = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x           = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1     = c1

    def forward(self, x):
        # bs, self.reg_max * 4, 8400
        b, c, a = x.shape
        # bs, 4, self.reg_max, 8400 => bs, self.reg_max, 4, 8400 => b, 4, 8400
        # 以softmax的方式，对0~16的数字计算百分比，获得最终数字。
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)
        
#---------------------------------------------------#
#   yolo_body
#---------------------------------------------------#
class YoloBody(nn.Module):
    def __init__(self, input_shape, num_classes, phi, pretrained=False):
        super(YoloBody, self).__init__()
        depth_dict          = {'n' : 0.33, 's' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.00,}
        width_dict          = {'n' : 0.25, 's' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}
        deep_width_dict     = {'n' : 1.00, 's' : 1.00, 'm' : 0.75, 'l' : 0.50, 'x' : 0.50,}
        dep_mul, wid_mul, deep_mul = depth_dict[phi], width_dict[phi], deep_width_dict[phi]

        base_channels       = int(wid_mul * 64)  # 64
        base_depth          = max(round(dep_mul * 3), 1)  # 3
        #-----------------------------------------------#
        #   输入图片是3, 640, 640
        #-----------------------------------------------#

        #---------------------------------------------------#   
        #   生成主干模型
        #   获得三个有效特征层，他们的shape分别是：
        #   256, 80, 80
        #   512, 40, 40
        #   1024 * deep_mul, 20, 20
        #---------------------------------------------------#
        self.backbone   = Backbone(base_channels, base_depth, deep_mul, phi, pretrained=pretrained)

        #------------------------加强特征提取网络------------------------# 
        self.upsample   = nn.Upsample(scale_factor=2, mode="nearest")

        # 1024 * deep_mul + 512, 40, 40 => 512, 40, 40
        self.conv3_for_upsample1    = C2f(int(base_channels * 16 * deep_mul) + base_channels * 8, base_channels * 8, base_depth, shortcut=False)
        #self.conv3_for_upsample1 = C2f_FasterBlock(int(base_channels * 16 * deep_mul) + base_channels * 8, base_channels * 8,shortcut=False)
        # 768, 80, 80 => 256, 80, 80
        self.conv3_for_upsample2    = C2f(base_channels * 8 + base_channels * 4, base_channels * 4, base_depth, shortcut=False)

        # 256, 80, 80 => 256, 40, 40
        self.down_sample1           = Conv(base_channels * 4, base_channels * 4, 3, 2)
        # 512 + 256, 40, 40 => 512, 40, 40
        self.conv3_for_downsample1  = C2f(base_channels * 8 + base_channels * 4, base_channels * 8, base_depth, shortcut=False)

        # 512, 40, 40 => 512, 20, 20
        self.down_sample2           = Conv(base_channels * 8, base_channels * 8, 3, 2)
        # 1024 * deep_mul + 512, 20, 20 =>  1024 * deep_mul, 20, 20
        self.conv3_for_downsample2  = C2f(int(base_channels * 16 * deep_mul) + base_channels * 8, int(base_channels * 16 * deep_mul), base_depth, shortcut=False)
        #=====================================================================================
        # TFF模块params=13.527M，Gflops= 33.819G
        #self.feat0_A                = Conv(base_channels * 2,base_channels * 4)
        # self.Adaptive_MaxPool_0     = nn.AdaptiveMaxPool2d((80,80))
        # self.Adaptive_AvgPool_0     = nn.AdaptiveAvgPool2d((80,80))
        # self.feat0_B                = Conv(base_channels * 4 , base_channels * 4)
        self.feat3_EMA = cbam_block(512)
        self.feat2_EMA = cbam_block(256)
        self.feat1_EMA = cbam_block(128)
        # #
        self.P3_A                   = SPPF(base_channels * 4 , base_channels * 4)
        self.P4_A                   = SPPF(base_channels * 8, base_channels * 8)
        self.P5_A                   = SPPF(base_channels * 16, base_channels * 16)
        self.P3                     = Conv(base_channels * 8 , base_channels * 4)
        self.feat_A                 = Conv(base_channels * 8, base_channels * 4)
        self.feat_B                 = Conv(base_channels * 4, base_channels * 4)
        self.feat1_A                = Conv(base_channels * 4,base_channels * 8)
        self.Adaptive_MaxPool_1     = nn.AdaptiveMaxPool2d((40,40))
        self.Adaptive_AvgPool_1     = nn.AdaptiveAvgPool2d((40,40))
        self.feat1_B                = Conv(base_channels * 8,base_channels * 8)
        self.P4                     = Conv(base_channels * 24, base_channels * 8)
        self.feat3_A                = Conv(base_channels * 16, base_channels * 8)
        self.feat3_B                = Conv(base_channels * 8, base_channels * 8)
        self.feat4_A                = Conv(base_channels * 8, base_channels * 16)
        self.Adaptive_MaxPool_2     = nn.AdaptiveMaxPool2d((20,20))
        self.Adaptive_AvgPool_2     = nn.AdaptiveAvgPool2d((20,20))
        self.feat4_B                = Conv(base_channels * 16, base_channels * 16)
        self.P5                     = Conv(base_channels * 32, base_channels * 16)
        self.P3_U                   = Conv(base_channels * 12, base_channels * 4)
#=====================================================================================
# #BiFPN结构,params=14.533M , Gflops = 33.150G
#         # self.Conv3 = Conv(int(base_channels * 16 * deep_mul) , base_channels * 8)
#         # self.BiFPN3 = BiFPN_Concat(base_channels * 8 , int(base_channels * 16 * deep_mul) + base_channels * 8)
#         self.P5_down = Conv(base_channels * 8, int(base_channels * 16 * deep_mul), 3, 2)
#         self.P5_Biconcat1 = BiFPN_Concat(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul))
#         self.P4_down = Conv(base_channels * 4, base_channels * 8, 3, 2)
#         self.P4_Biconcat1 = BiFPN_Concat(base_channels * 8, base_channels * 8)
#         self.P3_detect = BiFPN_Concat(base_channels * 4, base_channels * 4)
#         self.P3_dowmsample = Conv(base_channels * 4, base_channels * 8)
#         self.P4_Biconcat2 = BiFPN_Concat(base_channels * 8, base_channels * 8)
#         self.P4_detect = C2f(base_channels * 8, base_channels * 8, base_depth, shortcut=False)
#         self.P4_conv = Conv(base_channels * 8, base_channels * 4)
#         self.P4_downsample = Conv(base_channels * 8, int(base_channels * 16 * deep_mul))
#         self.P5_Biconcat2 = BiFPN_Concat(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul))
#         self.P5_detect = C2f(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), base_depth, shortcut=False)
#         self.P5_downsample = Conv(base_channels * 8, int(base_channels * 16 * deep_mul) , 3 , 2)
# =====================================================================================
#更新结构params=13.893M , Gflops = 32.520G
        # self.P5_down = Conv(base_channels * 8, int(base_channels * 16 * deep_mul), 3, 2)
        # self.P5_Biconcat1 = BiFPN_Concat(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul))
        # self.P5_Conv = Conv(int(base_channels * 16 * deep_mul), base_channels * 8)
        # self.P4_down = Conv(base_channels * 4, base_channels * 8, 3, 2)
        # self.P4_Biconcat1 = BiFPN_Concat(base_channels * 8 , base_channels * 8)
        # self.P4_med = C2f(base_channels * 8, base_channels * 8, base_depth, shortcut=False)
        # self.P4_Conv = Conv(base_channels * 8, base_channels * 4)
        # self.P3_Biconcat1 = BiFPN_Concat(base_channels * 4, base_channels * 4)
        # self.P3_detect = C2f(base_channels * 4, base_channels * 4, base_depth, shortcut=False)
        # self.P3_down_sample = Conv(base_channels * 4, base_channels * 8, 3, 2)
        # self.P4_endBiconcat = BiFPN_Concat(base_channels * 8, base_channels * 8)
        # self.P4_detect = C2f(base_channels * 8, base_channels * 8, base_depth, shortcut=False)
        # self.P4_down_sample = Conv(base_channels * 8, int(base_channels * 16 * deep_mul), 3, 2)
        # self.P5_endBiconcat1 = BiFPN_Concat(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul))
        # self.P5_detect = C2f(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), base_depth, shortcut=False)
# =====================================================================================
#         self.EMA3 = EMA(base_channels * 4)
#         self.EMA4 = EMA(base_channels * 8)
#         self.EMA5 = EMA(int(base_channels * 16 * deep_mul))
# =======================================================================
# # 更新结构，(P5,P4,P3进行向上融合),params=19.041M , Gflops = 34.852G
# # 在P6->P5添加DWConv+ELA,Params = 14.859M , Gflops = 34.015G
# #12.441M        ......
# #         self.P5_down = Conv(int(base_channels * 16 * deep_mul), int(base_channels * 32 * deep_mul),3,2)
#         self.P5_down = DWConv(int(base_channels * 16 * deep_mul), int(base_channels * 32 * deep_mul))
#         self.ELA = EfficientLocalizationAttention(int(base_channels * 32 * deep_mul))
#         # self.Bi_Former = BiLevelRoutingAttention(int(base_channels * 32 * deep_mul))
#         self.P6_up = Conv(int(base_channels * 32 * deep_mul),int(base_channels * 16 * deep_mul))
#         self.P5_Biconcat1 = BiFPN_Concat(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul))
#         self.P5_Conv = C2f(int(base_channels * 16 * deep_mul), base_channels * 8, base_depth, shortcut=False)
#         # self.P5_Conv = Conv(int(base_channels * 16 * deep_mul), base_channels * 8)
#         self.P5_up = Conv(int(base_channels * 16 * deep_mul), base_channels * 8)
#         self.P4_Biconcat1 = BiFPN_Concat(base_channels * 8, base_channels * 8)
#         self.P4_Conv = C2f(base_channels * 8, base_channels * 4, base_depth, shortcut=False)
#         self.P4_up = Conv(base_channels * 8, base_channels * 4)
#         self.P3_Biconcat1 = BiFPN_Concat(base_channels * 4, base_channels * 4)
#         self.P3_detect = C2f(base_channels * 4, base_channels * 4, base_depth, shortcut=False)
#         self.P3_down_sample = Conv(base_channels * 4, base_channels * 8, 3, 2)
#         # self.P3_down_sample = ADown(base_channels * 4, base_channels * 8)
#         # self.P3_down_sample = DWConv(base_channels * 4, base_channels * 8)
#         self.P4_endBiconcat = BiFPN_Concat(base_channels * 8, base_channels * 8)
#         self.P4_detect = C2f(base_channels * 8, base_channels * 8, base_depth, shortcut=False)
#         self.P4_down_sample = Conv(base_channels * 8, int(base_channels * 16 * deep_mul), 3, 2)
#         self.P5_down_sample = Conv(base_channels * 8, int(base_channels * 16 * deep_mul), 3, 2)
#         # self.P4_down_sample = ADown(base_channels * 8, int(base_channels * 16 * deep_mul))
#         # self.P5_down_sample = ADown(base_channels * 8, int(base_channels * 16 * deep_mul))
#         self.P5_endBiconcat = BiFPN_Concat(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul))
#         self.P5_detect = C2f(int(base_channels * 16 * deep_mul), int(base_channels * 16 * deep_mul), base_depth,shortcut=False)
# =======================================================================
        #------------------------加强特征提取网络------------------------#
        
        ch              = [base_channels * 4, base_channels * 8, int(base_channels * 16 * deep_mul)]
        self.shape      = None
        self.nl         = len(ch)
        # self.stride     = torch.zeros(self.nl)
        self.stride     = torch.tensor([256 / x.shape[-2] for x in self.backbone.forward(torch.zeros(1, 3, 256, 256))])  # forward
        self.reg_max    = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no         = num_classes + self.reg_max * 4  # number of outputs per anchor
        self.num_classes = num_classes
        
        c2, c3   = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], num_classes)  # channels
        # self.cv2 = nn.ModuleList(nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        # self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, num_classes, 1)) for x in ch)
        self.cv2 = nn.ModuleList(nn.Sequential(GSConv(x, c2), GSConv(c2, c2), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(nn.Sequential(GSConv(x, c3), GSConv(c3, c3 ), nn.Conv2d(c3, num_classes, 1)) for x in ch)
        if not pretrained:
            weights_init(self)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()


    def fuse(self):
        print('Fusing layers... ')
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        return self
    
    def forward(self, x):
        #  backbone
        feat1, feat2, feat3 = self.backbone.forward(x)
        # feat1 = self.ELA3(feat1)
        # feat2 = self.ELA4(feat2)
        # feat3 = self.ELA5(feat3)
#=======================================================================
        # # feat0_CBL       = self.upsample(feat1)
        # # feat0_MA        = self.Adaptive_MaxPool_0(feat0_CBL) + self.Adaptive_AvgPool_0(feat0_CBL)
        # # feat0_end       = self.feat0_B(feat0_MA)
        # feat1_CBL       = self.feat1_A(feat1)
        # feat1_MA        = self.Adaptive_MaxPool_1(feat1_CBL) + self.Adaptive_AvgPool_1(feat1_CBL)
        # feat1_end       = self.feat1_B(feat1_MA)
        # feat3_CBL       = self.feat3_A(feat3)
        # feat3_up        = self.upsample(feat3_CBL)
        # feat3_end       = self.feat3_B(feat3_up)
        # feat4_CBL       = self.feat4_A(feat2)
        # feat4_MA        = self.Adaptive_MaxPool_2(feat4_CBL) + self.Adaptive_AvgPool_2(feat4_CBL)
        # feat4_end       = self.feat4_B(feat4_MA)
#=======================================================================
        feat3 = self.feat3_EMA(feat3)
        feat2 = self.feat2_EMA(feat2)
        feat1 = self.feat1_EMA(feat1)

        #------------------------加强特征提取网络------------------------#
# =======================================================================
# # 添加BiFPN,params=14.533M , Gflops = 33.150G
#         P5_down = self.P5_down(feat2)
#         P5_Biconcat1 = self.P5_Biconcat1([P5_down , feat3])
#         P5_upsample = self.upsample(P5_Biconcat1)
#         P4 = torch.cat([P5_upsample, feat2], 1)
#         P4 = self.conv3_for_upsample1(P4)
#         P4_down = self.P4_down(feat1)
#         P4_Biconcat1 = self.P4_Biconcat1([P4_down, P4])
#         P4_upsample = self.upsample(P4_Biconcat1)
#         P3 = torch.cat([P4_upsample, feat1], 1)
#         P3 = self.conv3_for_upsample2(P3)
#         P4_upsample =self.P4_conv(P4_upsample)
#         P3 = self.P3_detect([P3 , P4_upsample])
#         P3_downsample = self.down_sample1(P3)
#         P3_downsample = self.P3_dowmsample(P3_downsample)
#         P4 = self.P4_Biconcat2([P3_downsample , P4_Biconcat1 , P4])
#         P4 = self.P4_detect(P4)
#         P4_downsample = self.down_sample2(P4)
#         P4_downsample = self.P4_downsample(P4_downsample)
#         P4_Biconcat1 = self.P5_downsample(P4_Biconcat1)
#         P5 = self.P5_Biconcat2([P4_downsample , P5_Biconcat1 , P4_Biconcat1])
#         P5 = self.P5_detect(P5)
# =======================================================================
#更新结构,params=13.893M , Gflops = 32.520G
        # P5_down = self.P5_down(feat2)
        # P5_Biconcat1 = self.P5_Biconcat1([P5_down , feat3])
        # P5_upsample = self.upsample(P5_Biconcat1)
        # P5_upsample = self.P5_Conv(P5_upsample)
        # P4_down = self.P4_down(feat1)
        # P4_Biconcat1 = self.P4_Biconcat1([feat2 , P5_upsample , P4_down])
        # P4_med = self.P4_med(P4_Biconcat1)
        # P4_upsample = self.upsample(P4_med)
        # P4_upsample = self.P4_Conv(P4_upsample)
        # P3_Biconcat1 = self.P3_Biconcat1([P4_upsample , feat1])
        # P3 = self.P3_detect(P3_Biconcat1)
        # P3_down = self.P3_down_sample(P3)
        # P4_concat = self.P4_endBiconcat([P3_down , P4_med])
        # P4 = self.P4_detect(P4_concat)
        # P4_down = self.P4_down_sample(P4)
        # P5_concat = self.P5_endBiconcat1([P4_down , P5_Biconcat1])
        # P5 = self.P5_detect(P5_concat)
# =======================================================================
# =======================================================================
#         # 更新结构，(P5,P4,P3进行向上融合),params=19.041M , Gflops = 34.852G
#         #在P6->P5添加DWConv+ELA,Params = 14.859M , Gflops = 34.015G
#         #12.441M，26.134G
#         P6 = self.P5_down(feat3)
#         P6_ELA = self.ELA(P6)
#         # P6_ELA = self.Bi_Former(P6)
#         P6_up = self.P6_up(self.upsample(P6_ELA))
#         # P6_up = self.P6_up(self.upsample(P6))
#         P5_td = self.P5_Biconcat1([P6_up , feat3])
#         P5_upsample = self.P5_Conv(self.upsample(P5_td))
#         P5_up = self.P5_up(self.upsample(feat3))
#         P4_td = self.P4_Biconcat1([feat2, P5_upsample, P5_up])
#         P4_upsample = self.P4_Conv(self.upsample(P4_td))
#         P4_up = self.P4_up(self.upsample(feat2))
#         P3_td = self.P3_Biconcat1([feat1, P4_upsample, P4_up])
#         P3 = self.P3_detect(P3_td)
#         P3_down = self.P3_down_sample(P3)
#         P4_out = self.P4_endBiconcat([P3_down , P4_td , P5_upsample])
#         P4 = self.P4_detect(P4_out)
#         P4_down = self.P4_down_sample(P4)
#         P4_td_down = self.P5_down_sample(P4_td)
#         P5_out = self.P5_endBiconcat([P4_down , P5_td , P4_td_down])
#         P5 = self.P5_detect(P5_out)
#         # P3 = self.EMA3(P3)
#         # P4 = self.EMA4(P4)
#         # P5 = self.EMA5(P5)
# =======================================================================
#YOLOv8s模块
        # 1024 * deep_mul, 20, 20 => 1024 * deep_mul, 40, 40
        P5_upsample = self.upsample(feat3)
        # 1024 * deep_mul, 40, 40 cat 512, 40, 40 => 1024 * deep_mul + 512, 40, 40
        P4          = torch.cat([P5_upsample, feat2], 1)
# =======================================================================
#添加BiFPN
        # P5_upsample = self.Conv3(P5_upsample)
        # P4   = self.BiFPN3([P5_upsample, feat2])
# =======================================================================
#YOLOv8s模块
        # 1024 * deep_mul + 512, 40, 40 => 512, 40, 40
        P4          = self.conv3_for_upsample1(P4)

        # 512, 40, 40 => 512, 80, 80
        P4_upsample = self.upsample(P4)
        # 512, 80, 80 cat 256, 80, 80 => 768, 80, 80
        P3          = torch.cat([P4_upsample, feat1], 1)
        # 768, 80, 80 => 256, 80, 80
        P3          = self.conv3_for_upsample2(P3)
# =======================================================================
        # 添加BiFPN
        # P5_upsample = self.Conv3(P5_upsample)
        # P4   = self.BiFPN3([P5_upsample, feat2])
# =======================================================================
        #==============================================================
        # P3_A       = torch.cat([P3, feat0_end], 1)
        # P3          = self.P3(P3_A)
        #==============================================================
#YOLOv8s模块
        # 256, 80, 80 => 256, 40, 40
        P3_downsample = self.down_sample1(P3)
        # 512, 40, 40 cat 256, 40, 40 => 768, 40, 40
        P4 = torch.cat([P3_downsample, P4], 1)
        # 768, 40, 40 => 512, 40, 40
        P4 = self.conv3_for_downsample1(P4)
        #==============================================================
        # P4_A  = torch.cat([P4 , feat1_end , feat3_end],1)
        # P4    = self.P4(P4_A)
        # P4_C  = self.feat_A(P4)
        # P4_C  = self.upsample(P4_C)
        # P4_C  = self.feat_B(P4_C)
        # # P3_A       = torch.cat([P3, feat0_end, P4_C], 1)
        # P3_A       =torch.cat([P3, P4_C], 1)
        # P3          = self.P3(P3_A)
        #==============================================================
# YOLOv8s模块
        # 512, 40, 40 => 512, 20, 20
        P4_downsample = self.down_sample2(P4)
        # 512, 20, 20 cat 1024 * deep_mul, 20, 20 => 1024 * deep_mul + 512, 20, 20
        P5 = torch.cat([P4_downsample, feat3], 1)
        # 1024 * deep_mul + 512, 20, 20 => 1024 * deep_mul, 20, 20
        P5 = self.conv3_for_downsample2(P5)
        #==============================================================
        P3 = self.P3_A(P3)
        P4 = self.P4_A(P4)
        P5 = self.P5_A(P5)
        #
        # P3 = self.feat1_EMA(P3)
        # P4 = self.feat2_EMA(P4)
        # P5 = self.feat3_EMA(P5)
        #
        feat1_CBL = self.feat1_A(P3)
        feat1_MA = self.Adaptive_MaxPool_1(feat1_CBL) + self.Adaptive_AvgPool_1(feat1_CBL)
        feat1_end = self.feat1_B(feat1_MA)
        feat3_CBL = self.feat3_A(P5)
        feat3_up = self.upsample(feat3_CBL)
        feat3_end = self.feat3_B(feat3_up)
        feat4_CBL = self.feat4_A(P4)
        feat4_MA = self.Adaptive_MaxPool_2(feat4_CBL) + self.Adaptive_AvgPool_2(feat4_CBL)
        feat4_end = self.feat4_B(feat4_MA)
        P4_C = self.feat_A(P4)
        P4_C = self.upsample(P4_C)
        P4_C = self.feat_B(P4_C)
        # P3_A       = torch.cat([P3, feat0_end, P4_C], 1)
        # P3   = self.P3_A(P3)
        P3_A = torch.cat([P3, P4_C], 1)
        P3 = self.P3(P3_A)
        # P4   = self.P4_A(P4)
        P4_A = torch.cat([P4, feat1_end, feat3_end], 1)
        P4 = self.P4(P4_A)
        #
        # # ==============================================================
        # # P5  = self.P5_A(P5)
        P5_C = torch.cat([P5, feat4_end], 1)
        P5 = self.P5(P5_C)
        #==============================================================
        # P5_U = self.upsample(P5)
        # P4 = torch.cat([P5_U, P4], 1)
        # P4 = self.P4(P4)
        # P4_U = self.upsample(P4)
        # P3 = torch.cat([P4_U, P3], 1)
        # P3 = self.P3_U(P3)
        # ==============================================================
        #------------------------加强特征提取网络------------------------# 
        # P3 256, 80, 80
        # P4 512, 40, 40
        # P5 1024 * deep_mul, 20, 20
        shape = P3.shape  # BCHW
        
        # P3 256, 80, 80 => num_classes + self.reg_max * 4, 80, 80
        # P4 512, 40, 40 => num_classes + self.reg_max * 4, 40, 40
        # P5 1024 * deep_mul, 20, 20 => num_classes + self.reg_max * 4, 20, 20
        x = [P3, P4, P5]
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)

        if self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape
        
        # num_classes + self.reg_max * 4 , 8400 =>  cls num_classes, 8400; 
        #                                           box self.reg_max * 4, 8400
        box, cls        = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2).split((self.reg_max * 4, self.num_classes), 1)
        # origin_cls      = [xi.split((self.reg_max * 4, self.num_classes), 1)[1] for xi in x]
        dbox            = self.dfl(box)
        return dbox, cls, x, self.anchors.to(dbox.device), self.strides.to(dbox.device)