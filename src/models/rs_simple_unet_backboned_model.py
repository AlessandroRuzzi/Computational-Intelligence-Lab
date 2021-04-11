from torch.optim import Optimizer

from src.models.architectures.unet_backboned import UNET
from src.models.rs_simple_base_model import RSSimpleBaseModel


class RSSimpleUNETBackbonedModel(RSSimpleBaseModel):
    def __init__(
        self,
        optimizer: Optimizer,
        in_channels: int = 3,
        out_channels: int = 1,
        dir_preds_test: str = "path",
        backbone_name: str = "resnet152",
        classes: int = 2,
    ) -> None:
        super().__init__(optimizer, in_channels, out_channels, dir_preds_test)

        self.model = UNET(backbone_name=backbone_name, classes=classes)
