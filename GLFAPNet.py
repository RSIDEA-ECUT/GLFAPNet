import torch
from torch import nn
from Model.My_Model.Swin_Transformer import BasicLayer
from torch.nn import functional as F
from torchsummary import summary
from thop import profile, clever_format


class FeatureInit(nn.Module):
    def __init__(self, nbands, out_channels):
        super(FeatureInit, self).__init__()
        self.conv_layer = nn.Conv2d(in_channels=nbands, out_channels=out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv_layer(x))
        return x


class MPC(nn.Module):
    def __init__(self, out_channels):
        super(MPC, self).__init__()
        self.branch1x1 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )

        self.branch7x7 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=7, padding=3),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        branch1x1_output = self.branch1x1(x)
        branch3x3_output = self.branch3x3(x)
        branch5x5_output = self.branch5x5(x)
        branch7x7_output = self.branch7x7(x)
        return torch.cat([branch1x1_output, branch3x3_output, branch5x5_output, branch7x7_output], dim=1)


class GLFA(nn.Module):
    def __init__(self, out_channels):
        super(GLFA, self).__init__()
        self.swin_layer = BasicLayer()
        self.middle_layer = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.mpc_layer = MPC(out_channels)
        self.trans_layer = nn.Conv2d(in_channels=4 * out_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        x = self.swin_layer(x)
        x = F.relu(self.middle_layer(x))
        x = self.mpc_layer(x)
        x = F.relu(self.trans_layer(x))
        return x


class FeatureExtract(nn.Module):
    def __init__(self, out_channels, basicblock_nums):
        super(FeatureExtract, self).__init__()
        self.feature_extract = nn.ModuleList([GLFA(out_channels) for i in range(basicblock_nums)])

    def forward(self, x):
        for i in range(len(self.feature_extract)):
            x = self.feature_extract[i](x)
        return x


class GLFAPNet(nn.Module):
    def __init__(self, nbands, basicblock_nums, out_channels):
        super(GLFAPNet, self).__init__()
        self.init_feature = FeatureInit(nbands, out_channels)
        self.feature_extract = FeatureExtract(out_channels, basicblock_nums)
        self.fused_conv = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.recon_conv = nn.Conv2d(in_channels=out_channels, out_channels=nbands - 1, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.init_feature(x)
        x2 = self.feature_extract(x1)
        x3 = self.fused_conv(x2)
        x4 = self.recon_conv(F.relu(x3 + x1))
        # return x4 + x[:, 0:4, :, :]
        return x4


if __name__ == "__main__":
    model = GLFAPNet(nbands=5, basicblock_nums=5, out_channels=32).cuda()
    x = torch.rand(1, 5, 256, 256).cuda()
    y = model(x)
    print(y.shape)

    print("===> Parameter numbers : %.3fM" % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    flops, params = profile(model, inputs=(x,))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)
