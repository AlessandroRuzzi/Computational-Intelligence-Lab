from .decoders.deeplabv3plus import DeepLabV3PlusDecoder 
from .encoders.efficientnet import EfficientNetEncoder

from torch import nn, Tensor

class EffLab(nn.Module):
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        encoder_name: str = 'efficientnet-b0',
        decoder_out_channels = 256
    ) -> None:
        super().__init__()
        
        self.encoder = EfficientNetEncoder(
            name=encoder_name,
            dilated=True
        )
        
        self.decoder = DeepLabV3PlusDecoder(
            in_channels = self.encoder.out_channels, 
            out_channels = decoder_out_channels, 
            dilations = [12, 24, 36]
        )

        self.head = nn.Sequential(
            nn.Conv2d(
                in_channels = decoder_out_channels,
                out_channels = 1,
                kernel_size = 1
            ),
            nn.UpsamplingBilinear2d(
                scale_factor = 4
            )
        )
        
    def forward(self, x: Tensor) -> Tensor:
        xs = self.encoder(x)
        x = self.decoder(xs)
        return self.head(x)
        