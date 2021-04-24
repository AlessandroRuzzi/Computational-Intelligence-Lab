from typing import Any, Dict, List, Sequence, Tuple, Union

import pytorch_lightning as pl
import torch
from torch.optim import Optimizer

import src.utils.model_utils as utils
from src.models.metrics.kaggle_accuracy import KaggleAccuracy

# from src.models.metrics.dice_loss import BinaryDiceLoss


class RSSimpleModel(pl.LightningModule):
    def __init__(
        self,
        loss: torch.nn.Module,
        architecture: torch.nn.Module,
        lr: float = 0.001,
        dir_preds_test: str = "path",
        use_scheduler: bool = False,
    ) -> None:
        super().__init__()

        self.model = architecture
        self.loss = loss
        self.lr = lr
        self.use_scheduler = use_scheduler
        self.save_hyperparameters()
        self.metrics = [("kaggle", KaggleAccuracy())]

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

    def training_step(
        self, batch: List[torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        loss, preds, targets = self.step(batch)
        preds_proba = torch.sigmoid(preds)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log_metrics("train", (preds_proba > 0.5), targets)

        return {"loss": loss}

    def validation_step(
        self, batch: List[torch.Tensor], batch_id: int
    ) -> Dict[str, torch.Tensor]:
        x, y = batch
        loss, preds, targets = self.step(batch)
        preds_proba = torch.sigmoid(preds)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log_metrics("val", (preds_proba > 0.5), targets)
        image_grid = utils.build_image_grid_train(x, y, preds_proba).cpu()
        self.logger[0].experiment.log_image(image_grid, "Validation Batch")

        return {"loss": loss}

    def test_step(self, batch: torch.Tensor, batch_id: int) -> Dict[str, torch.Tensor]:
        x, kaggle_ids = batch
        preds_proba = torch.sigmoid(self.forward(x))
        preds_discr = (preds_proba > 0.5).float()

        # Log prediction images
        image_grid = utils.build_image_grid_test(x, preds_proba).cpu()
        self.logger[0].experiment.log_image(image_grid, "Test Batch")

        # Save predictions images to folder
        utils.save_images(
            path=self.hparams["dir_preds_test"], ids=kaggle_ids, preds=preds_discr
        )

        return {}

    def on_test_end(self) -> None:
        # Generate csv from file
        submission_filename = "submission.csv"
        submission_file = utils.images_to_csv(
            path=self.hparams["dir_preds_test"],
            csv_filename=submission_filename,
            patch_size=11,
        )
        # Log subission file
        self.logger[0].experiment.log_asset(submission_file)

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Tuple[Sequence[Optimizer], Sequence[Any]]]:
        parameters = list(self.parameters())
        optimizer = torch.optim.Adam(
            list(filter(lambda p: p.requires_grad, parameters)), lr=self.lr
        )

        def lambda_scheduler(epoch: int) -> float:
            if self.use_scheduler:
                return 0.95 ** epoch
            else:
                return 1.0

        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=[lambda_scheduler]
            ),
            "name": "lr_scheduler",
            "interval": "epoch",
        }

        return [optimizer], [lr_scheduler]
