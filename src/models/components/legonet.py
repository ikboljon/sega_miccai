from typing import Tuple

import torch
import torch.nn as nn

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from typing import Union
from functools import partial
from torch.nn import functional as F


from src.models.components.legonet_layers import ux_block, LayerNorm
from src.models.components.layers import RESseNormConv3d
from src.models.components.layers import FastSmoothSeNormConv3d, RESseNormConv3d, UpConv
from src.models.components.legonet_encoder import LegoNetEncoderv1, LegoNetEncoderv2, LegoNetEncoderv3


class LegoNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        depths=[2,2,2,2],
        feature_size=[24,48,96,192],
        spatial_dims=3,
        hidden_size=768,
        reduction=2,
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        norm_name='instance',
        res_block=True,
        return_logits = False,

        ):
        super().__init__()
        
        self.return_logits = return_logits
        self.out_channels = out_channels
        n_filters = 24
        
        self.encoder = LegoNetEncoderv1(
            in_channels=in_channels,
            depths=depths,
            feature_size=feature_size,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
            hidden_size=hidden_size,
            reduction=reduction,
        )

        
        self.upconv_4 = nn.ConvTranspose3d(hidden_size, 8 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block_4_1_right = FastSmoothSeNormConv3d((8 + 8) * n_filters, 8 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_4_2_right = FastSmoothSeNormConv3d(8 * n_filters, 8 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.vision_4 = UpConv(8 * n_filters, n_filters, reduction, scale=8)

        self.upconv_3 = nn.ConvTranspose3d(8 * n_filters, 4 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block_3_1_right = FastSmoothSeNormConv3d((4 + 4) * n_filters, 4 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_3_2_right = FastSmoothSeNormConv3d(4 * n_filters, 4 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.vision_3 = UpConv(4 * n_filters, n_filters, reduction, scale=4)

        self.upconv_2 = nn.ConvTranspose3d(4 * n_filters, 2 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block_2_1_right = FastSmoothSeNormConv3d((2 + 2) * n_filters, 2 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_2_2_right = FastSmoothSeNormConv3d(2 * n_filters, 2 * n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.vision_2 = UpConv(2 * n_filters, n_filters, reduction, scale=2)

        self.upconv_1 = nn.ConvTranspose3d(2 * n_filters, 1 * n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block_1_1_right = FastSmoothSeNormConv3d((1 + 1) * n_filters, n_filters, reduction, kernel_size=3, stride=1, padding=1)
        self.block_1_2_right = FastSmoothSeNormConv3d(n_filters, n_filters, reduction, kernel_size=3, stride=1, padding=1)

        self.conv1x1 = nn.Conv3d(1 * n_filters, self.out_channels, kernel_size=1, stride=1, padding=0)


    def forward(self, x_in):
        outs = self.encoder(x_in)
        
        x = self.block_4_2_right(self.block_4_1_right(torch.cat([self.upconv_4(outs[4]), outs[3]], 1)))
        sv4 = self.vision_4(x)

        x = self.block_3_2_right(self.block_3_1_right(torch.cat([self.upconv_3(x), outs[2]], 1)))
        sv3 = self.vision_3(x)

        x = self.block_2_2_right(self.block_2_1_right(torch.cat([self.upconv_2(x), outs[1]], 1)))
        sv2 = self.vision_2(x)

        x = self.block_1_1_right(torch.cat([self.upconv_1(x), outs[0]], 1))
        x = x + sv4 + sv3 + sv2
        x = self.block_1_2_right(x)

        x = self.conv1x1(x)

        if self.return_logits:
            return x
        else:
            if self.out_channels == 1:
                return torch.sigmoid(x)
            else:
                return F.softmax(x, dim=1)