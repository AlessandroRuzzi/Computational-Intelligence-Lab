from typing import Any, Dict, List, Sequence, Tuple, Union

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from pytorch_lightning.metrics.classification import Accuracy
from torch.optim import Optimizer

from src.architectures.model import UNET

class RS_model(pl.LightningModule):
    def __init__(
        self,
        optimizer: Optimizer,
        input_size: int = 1 * 28 * 28,
        hidden_size: int = 128,
        output_size: int = 10,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = UNET(hparams=self.hparams)
        self.loss = torch.nn.CrossEntropyLoss()

        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()