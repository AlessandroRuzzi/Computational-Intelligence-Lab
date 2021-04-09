import pytorch_lightning as pl
from torch.optim import Optimizer


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
        self.freeze(modules=pl_module.get_encoder_params(), train_bn=self.train_bn)

    def finetune_function(
        self,
        pl_module: pl.LightningModule,
        epoch: int,
        optimizer: Optimizer,
        opt_idx: int,
    ) -> None:
        i = 0

        for milestone in self.milestones:
            if (
                epoch == milestone
                and milestone == self.milestones[len(self.milestones)]
            ):
                self.unfreeze_and_add_param_group(
                    modules=pl_module.get_encoder_params(),
                    optimizer=optimizer,
                    train_bn=self.train_bn,
                )
            elif epoch == milestone:
                self.unfreeze_and_add_param_group(
                    modules=pl_module.get_encoder_params()[: -self.layers[i]],
                    optimizer=optimizer,
                    train_bn=self.train_bn,
                )

            i += 1
