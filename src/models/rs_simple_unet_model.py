from torch.optim import Optimizer

from src.models.architectures.unet import UNET
from src.models.rs_simple_base_model import RSSimpleBaseModel


class RSSimpleUNETModel(RSSimpleBaseModel):
    def __init__(
        self, optimizer: Optimizer, in_channels: int = 3, out_channels: int = 1
    ) -> None:
        super().__init__(optimizer, in_channels, out_channels)

        self.model = UNET(in_channels=in_channels, out_channels=out_channels)
