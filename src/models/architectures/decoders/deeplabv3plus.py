from typing import List, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        padding: Union[int, Tuple[int, int]] = 0,
        stride: Union[int, Tuple[int, int]] = 1,
        dilation: Union[int, Tuple[int, int]] = 1,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=not use_batchnorm,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=out_channels)
            if use_batchnorm
            else nn.Identity(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class ConvSeparableBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        padding: Union[int, Tuple[int, int]] = 0,
        stride: Union[int, Tuple[int, int]] = 1,
        dilation: Union[int, Tuple[int, int]] = 1,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                dilation=dilation,
                groups=in_channels,
                bias=False,
            ),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=not use_batchnorm,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=out_channels)
            if use_batchnorm
            else nn.Identity(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class AvgPoolBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            ConvBlock(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        x = self.block(x)
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        return x


class ASPPBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilations: List[int],
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        branches = []
        # Simple convolution
        branches += [
            ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        ]
        branches += [
            ConvSeparableBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                dilation=dilation,
                padding=dilation,
            )
            for dilation in dilations
        ]
        # Pooled convolution
        branches += [AvgPoolBlock(in_channels=in_channels, out_channels=out_channels)]
        self.branches = nn.ModuleList(branches)

        self.head = nn.Sequential(
            ConvBlock(
                in_channels=(len(dilations) + 2) * out_channels,
                out_channels=out_channels,
                kernel_size=1,
            ),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        res = []
        for branch in self.branches:
            res.append(branch(x))
        res = torch.cat(res, dim=1)
        return self.head(res)


class DeepLabV3PlusDecoder(nn.Module):
    def __init__(
        self,
        in_channels: List[int],
        out_channels: int = 256,
        dilations: List[int] = [12, 24, 36],
    ) -> None:
        super().__init__()
        highres_out_channels = 48

        self.lowres = nn.Sequential(
            ASPPBlock(
                in_channels=in_channels[-1],
                out_channels=out_channels,
                dilations=dilations,
            ),
            ConvSeparableBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.UpsamplingBilinear2d(scale_factor=4),
        )

        self.highres = ConvBlock(
            in_channels=in_channels[-4],
            out_channels=highres_out_channels,
            kernel_size=1,
        )

        self.head = ConvSeparableBlock(
            in_channels=highres_out_channels + out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x: List[Tensor]) -> Tensor:
        lowres_features = self.lowres(x[-1])
        highres_features = self.highres(x[-4])
        features = torch.cat([lowres_features, highres_features], dim=1)
        return self.head(features)
