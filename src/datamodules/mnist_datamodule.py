from typing import Any

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms


class MNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 64,
        train_val_split: tuple = (55_000, 5_000),
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs: int,
    ) -> None:

        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_val_split = train_val_split
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        self.dims = (1, 28, 28)

    def prepare_data(self) -> None:
        # Download data
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: Any = None) -> None:
        # Transform and split datasets
        trainset = MNIST(self.data_dir, train=True, transform=self.transforms)
        testset = MNIST(self.data_dir, train=False, transform=self.transforms)
        self.data_train, self.data_val = random_split(trainset, self.train_val_split)
        self.data_test = testset

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
