from typing import List

from torch import nn, Tensor
from .encoders.resnet import ResNetEncoder
from .decoders.unet import UNetDecoder


class UNet(nn.Module): 
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1, 
        encoder_name: str = 'resnet101',
        encoder_depth: int = 5,
        encoder_pretrained: bool = True,
        decoder_channels: List[int] = [256, 128, 64, 32, 16],
        decoder_use_batchnorm: bool = True
    ) -> None:
        super().__init__() 
        
        self.encoder = ResNetEncoder(
            name = encoder_name,
            pretrained = encoder_pretrained,
            depth = encoder_depth
        )

        self.decoder = UNetDecoder(
            inputs_channels = self.encoder.out_channels[1:][::-1],
            blocks_channels = list(decoder_channels),
            n_blocks = len(decoder_channels),
            use_batchnorm = decoder_use_batchnorm
        )

        self.head = nn.Conv2d(
            in_channels = decoder_channels[-1],
            out_channels = out_channels,
            kernel_size = 3, 
            padding = 1
        )

    def forward(self, x: Tensor) -> Tensor:
        xs = self.encoder(x) 
        x = self.decoder(xs[1:][::-1])
        x = self.head(x)
        return x