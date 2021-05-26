from typing import List, Union, Tuple

import torch 
import torch.nn.functional as F
from torch import nn, Tensor

class ConvBlock(nn.Module):

    def __init__(
        self, 
        in_channels: int,
        out_channels: int, 
        kernel_size: Union[int, Tuple[int, int]], 
        padding: Union[int, Tuple[int, int]] = 0, 
        stride: Union[int, Tuple[int, int]] = 1, 
        dilation: Union[int, Tuple[int, int]] = 1, 
        use_batchnorm: bool = True
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride = stride,
                padding = padding,
                dilation = dilation,
                bias = not use_batchnorm
            ), 
            nn.ReLU(inplace = True), 
            nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity() 
        )  
    
    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__() 
        self.block = nn.Sequential(
            ConvBlock(
                in_channels + skip_channels,
                out_channels,
                kernel_size = 3,
                padding = 1,
                use_batchnorm = use_batchnorm
            ),
            ConvBlock(
                out_channels,
                out_channels,
                kernel_size = 3,
                padding = 1,
                use_batchnorm = use_batchnorm
            )
        )
    
    def forward(self, x: Tensor, skip: Tensor = None) -> Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        out = self.block(x)
        return out


class UNetDecoder(nn.Module):

    def __init__(
        self,
        inputs_channels: List[int],
        blocks_channels: List[int] = [256, 128, 64, 32, 16],
        n_blocks: int = 5,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__() 
       # assert n_blocks == len(inputs_channels)
        assert n_blocks == len(blocks_channels)
        self.n_blocks = n_blocks 

        in_channels_list = [inputs_channels[0]] + blocks_channels
        skip_channels_list = inputs_channels[1:] + [0]
        blocks = []

        for idx in range(n_blocks):
            blocks += [DecoderBlock(
                in_channels = in_channels_list[idx],
                skip_channels = skip_channels_list[idx],
                out_channels = blocks_channels[idx],
                use_batchnorm = use_batchnorm
            )]

        self.blocks = nn.ModuleList(blocks)

    def forward(self, inputs: List[Tensor]) -> Tensor:
        x = inputs[0]
        skips = inputs[1:] + [None]

        for idx in range(len(self.blocks)):
            x = self.blocks[idx](x, skips[idx])

        return x