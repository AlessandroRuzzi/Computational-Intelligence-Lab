import pytorch_lightning as pl
import torch
from torch.optim import Optimizer

# flake8: noqa


class StepFineTuning(pl.callbacks.finetuning.BaseFinetuning):
    def __init__(
        self,
        layers: tuple = (5, 30),
        milestones: tuple = (10, 20, 30),
        train_bn: bool = False,
    ) -> None:
        self.milestones = milestones
        self.layers = layers
        self.train_bn = train_bn

    def freeze_before_training(self, pl_module: pl.LightningModule) -> None:
        pl_module.freeze()

    def finetune_function(
        self,
        pl_module: pl.LightningModule,
        epoch: int,
        optimizer: Optimizer,
        opt_idx: int,
    ) -> None:
        for index, milestone in enumerate(self.milestones):
            if epoch == milestone and milestone == self.milestones[-1]:
                pl_module.unfreeze_encoder()
            elif epoch == milestone:
                pl_module.partial_unfreeze_encoder(self.layers[index])
