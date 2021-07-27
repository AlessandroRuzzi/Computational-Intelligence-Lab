from typing import Any, Dict, List, Sequence, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import Optimizer

import src.utils.model_utils as utils
from src.models.metrics.kaggle_accuracy import KaggleAccuracy
from src.models.rs_simple_model import RSSimpleModel

# flake8: noqa
# mypy: ignore-errors


class RSSensembleModel(pl.LightningModule):
    def __init__(
        self,
        loss: torch.nn.Module,
        architectures: str,
        lr: float = 0.001,
        dir_preds_test: str = "path",
        use_scheduler: bool = False,
    ) -> None:
        super().__init__()

        self.models = []
        for arc in architectures:
            model = RSSimpleModel()
            state_dict = torch.load(
                arc,
                map_location="cuda",
            )
            print(model.load_state_dict(state_dict))
            self.models.append(model)
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
        probs = []
        batch_size = x.shape[0]

        for index, model in enumerate(self.models):

            dtype = (
                torch.cuda.FloatTensor
                if torch.cuda.is_available()
                else torch.FloatTensor
            )
            # print("0: ",x.size())
            kernel_size = 400
            stride = kernel_size // 2
            x_unf = F.unfold(x, kernel_size, stride=stride)

            # print("1: ",x_unf.size())

            splits = x_unf.shape[2]

            x_unf = x_unf.permute(0, 2, 1)

            x_unf = x_unf.reshape(batch_size * splits, 3, 400, 400)

            # print("2: ",x_unf.size())
            splits_pred = []
            for split in range(splits):
                # print("SPLIT: ", split)
                # imshow(x_unf[batch_size * split : batch_size * split + batch_size, : , : , :][1, : , : , :])
                pred = torch.sigmoid(
                    model.forward(
                        x_unf[batch_size * split : batch_size * (split + 1), :, :, :]
                    )
                )
                splits_pred.append(pred)
                # print(pred.size())
            preds_proba = torch.cat(splits_pred, 0)
            # print("P_PROBA 0: ", preds_proba.size())
            preds_proba = preds_proba.reshape(batch_size, splits, 1 * 400 * 400)
            preds_proba = preds_proba.permute(0, 2, 1)
            # print("P_PROBA 1: ", preds_proba.size())
            pred_f = F.fold(preds_proba, x.shape[-2:], kernel_size, stride=stride)
            # print("PRED_f:", pred_f.size())
            # print("SHAPE: ", x.shape[-2:])
            norm_map = F.fold(
                F.unfold(
                    torch.ones(pred_f.size()).type(dtype), kernel_size, stride=stride
                ),
                x.shape[-2:],
                kernel_size,
                stride=stride,
            )
            # print("MAP: ", norm_map.shape)
            pred_f /= norm_map
            pred_f = torch.reshape(pred_f, (x.shape[0], 1, 608, 608))
            if index == 0:
                probs = pred_f
            else:
                probs = torch.cat((probs, pred_f), dim=1)

        preds_proba = torch.reshape(probs, (x.shape[0], len(self.models), 608, 608))
        preds_proba = torch.mean(preds_proba, dim=1)
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
