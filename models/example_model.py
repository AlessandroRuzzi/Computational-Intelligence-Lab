import os
from argparse import ArgumentParser, Namespace
from typing import Dict, List

# flake8: noqa
import comet_ml

import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets.mnist import MNIST


class SimpleClassifier(pl.LightningModule):
    def __init__(self, hidden_dim: int = 128, learning_rate: float = 1e-3) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.l1 = torch.nn.Linear(28 * 28, self.hparams.hidden_dim)
        self.l2 = torch.nn.Linear(self.hparams.hidden_dim, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return x

    def training_step(
        self, batch: List[torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return {"loss": loss}

    def validation_step(
        self, batch: List[torch.Tensor], batch_id: int
    ) -> Dict[str, torch.Tensor]:
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("valid_loss", loss)
        return {"loss": loss}

    def test_step(
        self, batch: List[torch.Tensor], batch_id: int
    ) -> Dict[str, torch.Tensor]:
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("test_loss", loss)
        return {"loss": loss}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


def main(args: Namespace) -> None:
    pl.seed_everything(1234)
    # ------------
    # data
    # ------------
    dataset = MNIST("", train=True, download=True, transform=transforms.ToTensor())
    mnist_test = MNIST("", train=False, download=True, transform=transforms.ToTensor())
    mnist_train, mnist_val = random_split(dataset, [55000, 5000])

    train_loader = DataLoader(mnist_train, batch_size=args.batch_size)
    val_loader = DataLoader(mnist_val, batch_size=args.batch_size)
    test_loader = DataLoader(mnist_test, batch_size=args.batch_size)

    # ------------
    # model
    # ------------
    model = SimpleClassifier(args.hidden_dim, args.learning_rate)

    # ------------
    # logger
    # ------------
    comet_logger = pl.loggers.CometLogger(
        api_key=os.getenv("comet_api_key", default=args.comet_api_key),
        workspace=os.getenv("comet_workspace", default=args.comet_workspace),
        save_dir=args.comet_log_dir,  # If no API key is given comet will log only to this directory.
    )

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args, logger=comet_logger)
    trainer.fit(model, train_loader, val_loader)

    # ------------
    # testing
    # ------------
    trainer.test(test_dataloaders=test_loader)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(
        parser
    )  # Adds arguments for the trainer (e.g. gpus etc.)
    parser.add_argument("--comet_api_key", default="", type=str)
    parser.add_argument("--comet_workspace", default="", type=str)
    parser.add_argument("--comet_log_dir", default="./comet_logs", type=str)
    parser.add_argument("--hidden_dim", default=128, type=int)
    parser.add_argument("--learning_rate", default=0.0001, type=float)
    parser.add_argument("--batch_size", default=32, type=int)
    args = parser.parse_args()
    main(args)
