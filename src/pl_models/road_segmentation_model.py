from typing import Any, Dict, List, Sequence, Tuple, Union

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from pytorch_lightning.metrics.classification import Accuracy
from torch.optim import Optimizer

from src.architectures.unet import UNET


class RoadSegmentationModel(pl.LightningModule):
    def __init__(
        self,
        optimizer: Optimizer,
        in_channels: int = 3,
        out_channels: int = 1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = UNET(
            in_channels=self.hparams.in_channels, out_channels=self.hparams.out_channels
        )
        self.loss = torch.nn.BCEWithLogitsLoss()

        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def step(self, batch: List[torch.Tensor]) -> tuple:
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        return loss, y_hat, y

    def training_step(
        self, batch: List[torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        loss, preds, targets = self.step(batch)
        # acc = self.train_accuracy(preds, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        # self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step(
        self, batch: List[torch.Tensor], batch_id: int
    ) -> Dict[str, torch.Tensor]:
        loss, preds, targets = self.step(batch)
        # acc = self.val_accuracy(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        # self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds, "targets": targets}

    def test_step(
        self, batch: List[torch.Tensor], batch_id: int
    ) -> Dict[str, torch.Tensor]:
        loss, preds, targets = self.step(batch)
        # acc = self.test_accuracy(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        # self.log("test/acc", acc, on_step=False, on_epoch=True)
        return {"loss": loss}

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Tuple[Sequence[Optimizer], Sequence[Any]]]:
        optim = instantiate(self.hparams.optimizer, params=self.parameters())
        return optim
