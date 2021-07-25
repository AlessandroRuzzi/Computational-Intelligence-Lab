from os import X_OK
from typing import Any, Dict, List, Sequence, Tuple, Union
import numpy as np
import pytorch_lightning as pl
import torch
from torch.optim import Optimizer
import torch.nn.functional as F

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

        batch_size = x.shape[0]

        dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        #print("0: ",x.size())
        kernel_size = 400
        stride = kernel_size//2 
        x_unf = F.unfold(x, kernel_size, stride=stride)

        #print("1: ",x_unf.size())

        splits = x_unf.shape[2]

        x_unf = x_unf.permute(0, 2, 1)
        
        x_unf = x_unf.reshape(batch_size * splits, 3, 400, 400)

        #print("2: ",x_unf.size())
        splits_pred = []
        for split in range(splits):
            #print("SPLIT: ", split)
            pred = torch.sigmoid(self.forward(x_unf[batch_size * split : batch_size * split + batch_size, : , : , :]))
            splits_pred.append(pred)
            #print(pred.size())

        preds_proba = torch.cat(splits_pred, 0)
        #print("P_PROBA: ", preds_proba.size())
        preds_proba = preds_proba.view(batch_size, 1 * 400 * 400, splits)

        pred_f = F.fold(preds_proba,x.shape[-2:],kernel_size,stride=stride)
        #print("PRED_f:", pred_f.size())
        norm_map = F.fold(F.unfold(torch.ones(pred_f.size()).type(dtype),kernel_size,stride=stride),x.shape[-2:],kernel_size,stride=stride)
        pred_f /= norm_map
        preds_discr = (pred_f > 0.5).float()
        
        # Log prediction images
        image_grid = utils.build_image_grid_test(x, pred_f).cpu()
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
            path=self.hparams["dir_preds_test"], csv_filename=submission_filename
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
