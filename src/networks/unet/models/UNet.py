"""UNet.py

Implements the UNet model for image segmentation
"""
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F 


def config_model(bn_config: int) -> Tuple[dict, str]:
    
    model_config_kwargs = {}
    if bn_config == 0: 
        model_config_kwargs['omit_bn_entry'] = False
        model_config_kwargs['omit_bn_up'] = False
        model_config_kwargs['omit_bn_down'] = False
        run_id_str = "all_bn_on"
    else: 
        raise ValueError(f'Only first configuration of batch norms is currently accepted. Received {bn_config}; Expected 0')

    return model_config_kwargs, run_id_str


class UNetNew(nn.Module):
    """UNetNew

    Referenced from: 
    https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
    
    """

    def __init__(self, n_channels, n_classes, bilinear=True, 
                 omit_bn_entry = False,
                 omit_bn_up = False, 
                 omit_bn_down = False, 
                 **kwargs):
        super(UNetNew, self).__init__()

        if omit_bn_entry or omit_bn_up or omit_bn_down: 
            print('OMITTING BATCH NORM: ')
            print(f'Inc double conv: {omit_bn_entry}, Downs: {omit_bn_down}, Ups: {omit_bn_up}\n')

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64, omit_bn = omit_bn_entry)
        self.down1 = Down(64, 128, omit_bn=omit_bn_down)
        self.down2 = Down(128, 256, omit_bn=omit_bn_down)
        self.down3 = Down(256, 512, omit_bn=omit_bn_down)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, omit_bn=omit_bn_down)
        self.up1 = Up(1024, 512 // factor, bilinear, omit_bn=omit_bn_up)
        self.up2 = Up(512, 256 // factor, bilinear, omit_bn=omit_bn_up)
        self.up3 = Up(256, 128 // factor, bilinear, omit_bn=omit_bn_up)
        self.up4 = Up(128, 64, bilinear, omit_bn=omit_bn_up)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, omit_bn=False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        if omit_bn: 
                self.double_conv = nn.Sequential(
                    nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                    # nn.BatchNorm2d(mid_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                    # nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
        else: 
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class DoubleAtrous(nn.Module): 
    """(atrous convolution => [BN] => ReLU ) * 2 """

    def __init__(self, in_channels, out_channels, mid_channels=None, omit_bn=False, dilation=2) -> None:
        super().__init__()
        if not mid_channels:
            mid_channels = in_channels
        if omit_bn: 
                self.double_conv = nn.Sequential(
                    nn.Conv2d(in_channels, mid_channels, kernel_size=3, dilation=dilation, padding='same'),
                    # nn.BatchNorm2d(mid_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(mid_channels, out_channels, kernel_size=3, dilation=dilation, padding='same'),
                    # nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
        else: 
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, dilation=dilation, padding='same'),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, dilation=dilation, padding='same'),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class BottleNeck(nn.Module): 
    """ 1x1 convs for compressing # channels"""

    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.bottleneck = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding='same')
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return self.bottleneck(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, omit_bn=False):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels, omit_bn=omit_bn)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, omit_bn=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, omit_bn=omit_bn)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels, omit_bn=omit_bn)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class MCMC_OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        """__init__

        Multichannel Multiclass OutConv block

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
        """
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):

        # reshape x
        return self.conv(x)


class SeqUNetFrameConv(nn.Module):

    def __init__(self, n_input_channels=2, n_classes=2, 
                 bottleneck_dim=200,
                 bilinear=True, 
                 verbose=False, 
                 **kwargs) -> None:

        super().__init__()

        self.verbose = verbose

        # convolutional settings
        self.n_channels = n_input_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # encoding branch
        self.inc = DoubleConv(n_input_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # self.bneck = BottleNeck(512, bottleneck_dim)
        self.projx4 = DoubleAtrous(512* 5, 512 * 5, mid_channels=bottleneck_dim*5)

        # decoding branch
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        if self.verbose: 
            print('SeqAttendNet: forward(): ')
            print(x.shape)
        nb, n_ang, _, h, w = x.shape

        x1s = []
        x2s = []
        x3s = []
        x4s = []
        x5s = []
        for ang_i in range(n_ang): 
            x1 = self.inc(x[:, ang_i, ...])        # type: torch.Tensor
            x2 = self.down1(x1)                    # type: torch.Tensor
            x3 = self.down2(x2)                    # type: torch.Tensor
            x4 = self.down3(x3)                    # type: torch.Tensor
            x5 = self.down4(x4)                    # type: torch.Tensor

            x1s.append(x1)
            x2s.append(x2)
            x3s.append(x3)
            x4s.append(x4)
            x5s.append(x5)
        
        x4cat = torch.cat(x4s, dim=1) # shape: [nb, 512 * 5, fullh//8, fullw//8]
        x4s = self.projx4(x4cat) # type: torch.Tensor #shape   [nb, 512 * 5, fullh//8, fullw//8]
        x4s = torch.tensor_split(x4s, n_ang, dim=1)

        # decoding branch
        all_logits = []
        for ang_i, (x1, x2, x3, x4, x5) in enumerate(zip(x1s, x2s, x3s, x4s, x5s)): 
            
            # concatenateo n the attn 
            # x5 = torch.cat([x5, cur_attn], dim=1) # type: torch.Tensor
            # x5 shape: [b, 1024 + self.full_dims, *self.pre_patch_sz]
            x = self.up1(x5, x4)                  # type: torch.Tensor
            # x4 shape: [b, 512, someh, somew]
            x = self.up2(x, x3)                   # type: torch.Tensor  
            x = self.up3(x, x2)                   # type: torch.Tensor  
            x = self.up4(x, x1)                   # type: torch.Tensor
            logits = self.outc(x)                 # type: torch.Tensor  
            all_logits.append(logits) # [B, 1, 2, H, W]
        
        all_logits = torch.stack(all_logits, dim=1)
        if self.verbose: 
            print('all_logits shape', all_logits.shape)

        return all_logits

