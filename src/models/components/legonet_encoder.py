from typing import Tuple

import torch
import torch.nn as nn
from typing import Optional, Sequence, Tuple, Type, Union

from timm.models.layers import trunc_normal_, DropPath

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from typing import Union
from functools import partial
from torch.nn import functional as F
from einops import rearrange
from monai.utils import ensure_tuple_rep


from src.models.components.layers import RESseNormConv3d
from src.models.components.layers import FastSmoothSeNormConv3d, RESseNormConv3d, UpConv
from src.models.components.legonet_layers import ux_block, LayerNorm

from monai.networks.blocks.transformerblock import TransformerBlock
from monai.networks.blocks.patchembedding import PatchEmbeddingBlock

from monai.networks.nets.swin_unetr import PatchMerging, PatchMergingV2, BasicLayer
from monai.utils import ensure_tuple_rep, look_up_option, optional_import
MERGING_MODE = {"merging": PatchMerging, "mergingv2": PatchMergingV2}


class LegoNetEncoderv1(nn.Module):
    def __init__(
        self,
        in_channels=1,
        depths=[2,2,2,2],
        feature_size=[24,48,96,192],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        hidden_size=768,
        reduction=2,

        ):
        super().__init__()

        self.downsample_layers = nn.ModuleList()

        # # UX stem 
        # self.stem = nn.Sequential(
        #     nn.Conv3d(in_channels, feature_size[0], kernel_size=7, stride=1, padding=3),
        #     LayerNorm(feature_size[0], eps=1e-6, data_format='channels_first')
        # )

        # SE stem
        self.stem = nn.Sequential(
            RESseNormConv3d(in_channels, feature_size[0], reduction, kernel_size=7, stride=1, padding=3),
            RESseNormConv3d(feature_size[0], feature_size[0], reduction, kernel_size=3, stride=1, padding=1)
        )


        # block 1 - UX
        self.block1_down = nn.Sequential(
            LayerNorm(feature_size[0], eps=1e-6, data_format='channels_first'),
            nn.Conv3d(feature_size[0], feature_size[1], kernel_size=2, stride=2)
        )
        
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 2
        self.block1_stage = nn.Sequential(
            *[ux_block(feature_size[1], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[1])]
        )

        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        layer = norm_layer(feature_size[1])
        layer_name = f'norm{1}'
        self.add_module(layer_name, layer)

        # block 2 - SE
        self.block2 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            RESseNormConv3d(feature_size[1], feature_size[2], reduction, kernel_size=3, stride=1, padding=1),
            RESseNormConv3d(feature_size[2], feature_size[2], reduction, kernel_size=3, stride=1, padding=1),
            RESseNormConv3d(feature_size[2], feature_size[2], reduction, kernel_size=3, stride=1, padding=1)
        )

        # block 3 - UX
        self.block3_down = nn.Sequential(
            LayerNorm(feature_size[2], eps=1e-6, data_format='channels_first'),
            nn.Conv3d(feature_size[2], feature_size[3], kernel_size=2, stride=2)
        )
        
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 6
        self.block3_stage = nn.Sequential(
            *[ux_block(feature_size[3], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[3])]
        )

        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        layer = norm_layer(feature_size[3])
        layer_name = f'norm{3}'
        self.add_module(layer_name, layer)

        # block 4 - SE
        self.block4 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            RESseNormConv3d(feature_size[3], hidden_size, reduction, kernel_size=3, stride=1, padding=1),
            RESseNormConv3d(hidden_size, hidden_size, reduction, kernel_size=3, stride=1, padding=1),
            RESseNormConv3d(hidden_size, hidden_size, reduction, kernel_size=3, stride=1, padding=1)
        )

    def forward_features(self, x_in):
        outs = []
        
        # stem
        x_stem = self.stem(x_in)
        outs.append(x_stem)

        # block 1
        x = self.block1_down(x_stem)
        x = self.block1_stage(x)
        norm_layer = getattr(self, f'norm{1}')
        x = norm_layer(x)
        outs.append(x)

        # block 2 
        x = self.block2(x)
        outs.append(x)

        # block 3 
        x = self.block3_down(x)
        x = self.block3_stage(x)
        norm_layer = getattr(self, f'norm{3}')
        x = norm_layer(x)
        outs.append(x)

        # block 4
        x = self.block4(x)
        outs.append(x)

        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)

        return x


class LegoNetEncoderv2(nn.Module):
    def __init__(
        self,
        in_channels=1,
        depths=[2,2,2,2],
        feature_size=[24,48,96,192],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        hidden_size=768,
        reduction=2,
        num_heads: Sequence[int] = (3, 6, 12, 24),
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        downsample="merging",
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        norm_layer: Type[LayerNorm] = nn.LayerNorm,
        ):
        super().__init__()

        self.downsample_layers = nn.ModuleList()
        self.patch_size = ensure_tuple_rep(16, spatial_dims)
        

        # # UX stem 
        # self.stem = nn.Sequential(
        #     nn.Conv3d(in_channels, feature_size[0], kernel_size=7, stride=1, padding=3),
        #     LayerNorm(feature_size[0], eps=1e-6, data_format='channels_first')
        # )

        # SE stem
        self.stem = nn.Sequential(
            RESseNormConv3d(in_channels, feature_size[0], reduction, kernel_size=7, stride=1, padding=3),
            RESseNormConv3d(feature_size[0], feature_size[0], reduction, kernel_size=3, stride=1, padding=1)
        )


        # block 1 - Swin
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        down_sample_mod = look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample
        self.window_size = ensure_tuple_rep(7, spatial_dims)
        self.block1_down = BasicLayer(
                dim=int(feature_size[0]),
                depth=depths[0],
                num_heads=num_heads[0],
                window_size=self.window_size,
                drop_path=dpr[sum(depths[:0]) : sum(depths[: 0 + 1])],
                mlp_ratio=4.0,
                qkv_bias=True,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                downsample=down_sample_mod,
                use_checkpoint=use_checkpoint,
            )
            

        # block 2 -  SE
        self.block2 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            RESseNormConv3d(feature_size[1], feature_size[2], reduction, kernel_size=3, stride=1, padding=1),
            RESseNormConv3d(feature_size[2], feature_size[2], reduction, kernel_size=3, stride=1, padding=1),
            RESseNormConv3d(feature_size[2], feature_size[2], reduction, kernel_size=3, stride=1, padding=1)
        )

        # block 3 - Swin
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        down_sample_mod = look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample
        self.window_size = ensure_tuple_rep(7, spatial_dims)
        self.block3_down = BasicLayer(
                dim=int(feature_size[2]),
                depth=depths[2],
                num_heads=num_heads[2],
                window_size=self.window_size,
                drop_path=dpr[sum(depths[:2]) : sum(depths[: 2 + 1])],
                mlp_ratio=4.0,
                qkv_bias=True,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                downsample=down_sample_mod,
                use_checkpoint=use_checkpoint,
            )
            

        # block 4 - SE
        self.block4 = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            RESseNormConv3d(feature_size[3], hidden_size, reduction, kernel_size=3, stride=1, padding=1),
            RESseNormConv3d(hidden_size, hidden_size, reduction, kernel_size=3, stride=1, padding=1),
            RESseNormConv3d(hidden_size, hidden_size, reduction, kernel_size=3, stride=1, padding=1)
        )


        
    def proj_out(self, x, normalize=False):
        if normalize:
            x_shape = x.size()
            if len(x_shape) == 5:
                n, ch, d, h, w = x_shape
                x = rearrange(x, "n c d h w -> n d h w c")
                x = F.layer_norm(x, [ch])
                x = rearrange(x, "n d h w c -> n c d h w")
            elif len(x_shape) == 4:
                n, ch, h, w = x_shape
                x = rearrange(x, "n c h w -> n h w c")
                x = F.layer_norm(x, [ch])
                x = rearrange(x, "n h w c -> n c h w")
        return x

    def forward_features(self, x_in):
        outs = []
        
        # stem
        x_stem = self.stem(x_in)
        outs.append(x_stem)

        # block 1
        # x = self.patch_embedding(x_stem)
        x = self.proj_out(x_stem, normalize=True)
        x = self.block1_down(x)
        # x = self.deproj_feat(x)
        outs.append(x)

        # block 2 
        x = self.block2(x)
        outs.append(x)

        # block 3 
        x = self.proj_out(x, normalize=True)
        x = self.block3_down(x)
        outs.append(x)

        # block 4
        x = self.block4(x)
        outs.append(x)

        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)

        return x


class LegoNetEncoderv3(nn.Module):
    def __init__(
        self,
        in_channels=1,
        depths=[2,2,2,2],
        feature_size=[24,48,96,192],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        hidden_size=768,
        reduction=2,
        num_heads: Sequence[int] = (3, 6, 12, 24),
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        downsample="merging",
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        norm_layer: Type[LayerNorm] = nn.LayerNorm,
        ):
        super().__init__()

        self.downsample_layers = nn.ModuleList()
        self.patch_size = ensure_tuple_rep(16, spatial_dims)
        

        # # UX stem 
        # self.stem = nn.Sequential(
        #     nn.Conv3d(in_channels, feature_size[0], kernel_size=7, stride=1, padding=3),
        #     LayerNorm(feature_size[0], eps=1e-6, data_format='channels_first')
        # )

        # SE stem
        self.stem = nn.Sequential(
            RESseNormConv3d(in_channels, feature_size[0], reduction, kernel_size=7, stride=1, padding=3),
            RESseNormConv3d(feature_size[0], feature_size[0], reduction, kernel_size=3, stride=1, padding=1)
        )


        # block 1 - Swin
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        down_sample_mod = look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample
        self.window_size = ensure_tuple_rep(7, spatial_dims)
        self.block1_down = BasicLayer(
                dim=int(feature_size[0]),
                depth=depths[0],
                num_heads=num_heads[0],
                window_size=self.window_size,
                drop_path=dpr[sum(depths[:0]) : sum(depths[: 0 + 1])],
                mlp_ratio=4.0,
                qkv_bias=True,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                downsample=down_sample_mod,
                use_checkpoint=use_checkpoint,
            )
            

        # block 2 - UX
        self.block2_down = nn.Sequential(
            LayerNorm(feature_size[1], eps=1e-6, data_format='channels_first'),
            nn.Conv3d(feature_size[1], feature_size[2], kernel_size=2, stride=2)
        )
        
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 4
        self.block2_stage = nn.Sequential(
            *[ux_block(feature_size[2], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[1])]
        )

        partial_norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        layer = partial_norm_layer(feature_size[2])
        layer_name = f'norm{2}'
        self.add_module(layer_name, layer)


        # block 3 - Swin
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        down_sample_mod = look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample
        self.window_size = ensure_tuple_rep(7, spatial_dims)
        self.block3_down = BasicLayer(
                dim=int(feature_size[2]),
                depth=depths[2],
                num_heads=num_heads[2],
                window_size=self.window_size,
                drop_path=dpr[sum(depths[:2]) : sum(depths[: 2 + 1])],
                mlp_ratio=4.0,
                qkv_bias=True,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                downsample=down_sample_mod,
                use_checkpoint=use_checkpoint,
            )
            

        # block 4 - UX
        self.block4_down = nn.Sequential(
            LayerNorm(feature_size[3], eps=1e-6, data_format='channels_first'),
            nn.Conv3d(feature_size[3], hidden_size, kernel_size=2, stride=2)
        )
        
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 6
        self.block4_stage = nn.Sequential(
            *[ux_block(hidden_size, drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[3])]
        )

        partial_norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        layer = partial_norm_layer(hidden_size)
        layer_name = f'norm{4}'
        self.add_module(layer_name, layer)



        
    def proj_out(self, x, normalize=False):
        if normalize:
            x_shape = x.size()
            if len(x_shape) == 5:
                n, ch, d, h, w = x_shape
                x = rearrange(x, "n c d h w -> n d h w c")
                x = F.layer_norm(x, [ch])
                x = rearrange(x, "n d h w c -> n c d h w")
            elif len(x_shape) == 4:
                n, ch, h, w = x_shape
                x = rearrange(x, "n c h w -> n h w c")
                x = F.layer_norm(x, [ch])
                x = rearrange(x, "n h w c -> n c h w")
        return x

    def forward_features(self, x_in):
        outs = []
        
        # stem
        x_stem = self.stem(x_in)
        outs.append(x_stem)

        # block 1
        # x = self.patch_embedding(x_stem)
        x = self.proj_out(x_stem, normalize=True)
        x = self.block1_down(x)
        # x = self.deproj_feat(x)
        outs.append(x)

        # block 2 
        x = self.block2_down(x)
        x = self.block2_stage(x)
        norm_layer = getattr(self, f'norm{2}')
        x = norm_layer(x)
        outs.append(x)


        # block 3 
        x = self.proj_out(x, normalize=True)
        x = self.block3_down(x)
        outs.append(x)

        # block 4
        x = self.block4_down(x)
        x = self.block4_stage(x)
        norm_layer = getattr(self, f'norm{4}')
        x = norm_layer(x)
        outs.append(x)

        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)

        return x
