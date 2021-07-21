from typing import Any, Dict, List, Sequence, Tuple, Union
import numpy as np
import pytorch_lightning as pl
import torch
from torch.optim import Optimizer

import src.utils.model_utils as utils
from src.models.metrics.kaggle_accuracy import KaggleAccuracy

# from src.models.metrics.dice_loss import BinaryDiceLoss

def split_picture(test_picture):
    split_1 = test_picture[:400, :400]
    split_2 = test_picture[:400, -400:]
    split_3 = test_picture[-400:, :400]
    split_4 = test_picture[-400:, -400:]
    return [split_1, split_2, split_3, split_4]


def merge_splits(split_1, split_2, split_3, split_4, mode='mean'):
    assert mode in ['mean', 'max'], 'mode can only be one between mean and max!'
    if mode == 'mean':
        function = np.mean
    elif mode == 'max':
        function = np.max
    intersect1 = function(np.array([split_1[:208, 208:], split_2[:208, :192]]), axis=0)
    intersect2 = function(np.array([split_3[192:, 208:], split_4[192:, :192]]), axis=0)

    intersect3 = function(np.array([split_1[208:, :208], split_3[:192, :208]]), axis=0)
    intersect4 = function(np.array([split_2[208:, 192:], split_4[:192, 192:]]), axis=0)

    intersect5 = function(np.array([split_1[208:, 208:], 
                                split_2[208:, :192],
                                split_3[:192, 208:],
                                split_4[:192, :192]]), axis=0)

    north_west = split_1[:208, :208]
    north_east = split_2[:208, 192:]
    south_west = split_3[192:, :208]
    south_east = split_4[192:, 192:]

    upper_slice = np.concatenate([north_west, intersect1, north_east], axis= 1)
    middle_slice = np.concatenate([intersect3, intersect5, intersect4], axis= 1)
    lower_slice = np.concatenate([south_west, intersect2, south_east], axis= 1)

    final_image = np.concatenate([upper_slice, middle_slice, lower_slice], axis= 0)
    return final_image

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

        print("X SHAPE: " , x.shape)

        lst1 = []
        lst2 = []
        for i in range(0, x.shape[0]):
            split_1, split_2, split_3, split_4 = split_picture(x[i,:, :])
            preds_proba1 = torch.sigmoid(self.forward(split_1)).numpy()
            preds_proba2 = torch.sigmoid(self.forward(split_2)).numpy()
            preds_proba3 = torch.sigmoid(self.forward(split_3)).numpy()
            preds_proba4 = torch.sigmoid(self.forward(split_4)).numpy()

            preds_proba = torch.from_numpy(merge_splits(preds_proba1, preds_proba2, preds_proba3, preds_proba4, mode = 'max'))
            preds_discr = (preds_proba > 0.5).float()

            lst1.append(preds_proba)
            lst2.append(preds_discr)
        

        preds_proba = torch.stack(lst1)
        preds_discr = torch.stack(lst2)
        
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
