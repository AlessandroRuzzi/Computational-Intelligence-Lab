import os
from typing import Any, Dict, List, Sequence, Tuple, Union

import pytorch_lightning as pl
import torch
import torchvision
from hydra.utils import instantiate
from torch.optim import Optimizer
from torchvision.utils import save_image

from src.models.metrics.dice_loss import BinaryDiceLoss
from src.models.metrics.kaggle_accuracy import KaggleAccuracy


class RSSimpleBaseModel(pl.LightningModule):
    def __init__(
        self,
        optimizer: Optimizer,
        in_channels: int = 3,
        out_channels: int = 1,
        dir_preds_test: str = "path",
    ) -> None:
        super().__init__()

        self.save_hyperparameters()
        self.model = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.loss = torch.nn.BCEWithLogitsLoss()

        self.metrics = [("kaggle", KaggleAccuracy()), ("dice", BinaryDiceLoss())]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def step(self, batch: List[torch.Tensor]) -> tuple:
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        return loss, y_hat, y

    def log_metrics(
        self, stage: "str", preds: torch.Tensor, targets: torch.Tensor
    ) -> None:
        for metric_name, metric in self.metrics:
            self.log(
                f"{stage}_{metric_name}",
                metric(preds, targets),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )

    def log_images(
        self, title: str, x: torch.Tensor, y: torch.Tensor, preds: torch.Tensor
    ) -> None:
        batch_x = x
        batch_y = y.expand(-1, 3, -1, -1)
        # batch_preds = preds.expand(-1, 3, -1, -1)
        batch_preds_sigmoid = torch.sigmoid(preds).expand(-1, 3, -1, -1)
        batch_preds_sigmoid_treshold = (torch.sigmoid(preds) > 0.5).expand(
            -1, 3, -1, -1
        )

        block = torch.cat(
            (
                batch_x,
                batch_y,
                batch_preds_sigmoid,
                batch_preds_sigmoid_treshold,
            ),
            0,
        )

        img_grid = torchvision.utils.make_grid(block)
        img_grid = img_grid.permute((1, 2, 0))

        self.logger[0].experiment.log_image(img_grid.cpu(), title)

    def training_step(
        self, batch: List[torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        loss, preds, targets = self.step(batch)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log_metrics("train", (torch.sigmoid(preds) > 0.5), targets)

        return {"loss": loss}

    def validation_step(
        self, batch: List[torch.Tensor], batch_id: int
    ) -> Dict[str, torch.Tensor]:
        x, y = batch
        loss, preds, targets = self.step(batch)
        # acc = self.val_accuracy(preds, targets)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log_metrics("val", (torch.sigmoid(preds) > 0.5), targets)
        self.log_images("Validation Batch", x, y, preds)

        return {"loss": loss}

    def test_step(self, batch: torch.Tensor, batch_id: int) -> Dict[str, torch.Tensor]:
        x, kaggle_ids = batch
        preds = (torch.sigmoid(self.forward(x)) > 0.5).float()

        # Save predictions images to folder
        os.makedirs(self.hparams["dir_preds_test"], exist_ok=True)

        for i in range(x.shape[0]):
            save_image(
                preds[i],
                os.path.join(
                    self.hparams["dir_preds_test"], f"satImage_{kaggle_ids[i]}.png"
                ),
            )

        return {}

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Tuple[Sequence[Optimizer], Sequence[Any]]]:
        optim = instantiate(self.hparams.optimizer, params=self.parameters())
        return optim
