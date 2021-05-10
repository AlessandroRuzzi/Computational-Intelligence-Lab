from typing import Any, Callable, Optional

import pytorch_lightning as pl
import torchvision
from torch.utils.data import DataLoader, random_split

from src.datamodules.datasets.google_maps_dataset import GoogleMapsDataset
from src.datamodules.datasets.rs_kaggle_dataset import RSKaggleDataset
from src.datamodules.transforms.to_tensor import ToTensor


class RSSimpleDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        google_maps_api: str = "",
        batch_size: int = 64,
        transforms_train: Callable = ToTensor(),
        transforms_test: Callable = torchvision.transforms.ToTensor(),
        train_val_split: tuple = (80, 20),
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs: int,
    ) -> None:

        super().__init__()

        self.data_dir = data_dir
        self.google_maps_api = google_maps_api
        self.batch_size = batch_size
        self.train_val_split = train_val_split
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.transforms_train = transforms_train
        self.transforms_test = transforms_test

        self.data_pretrain: Any = None
        self.data_train: Any = None
        self.data_val: Any = None
        self.data_test: Any = None

    def get_transforms(self) -> Any:
        return self.transforms_train

    def prepare_data(self) -> None:
        # Download data
        RSKaggleDataset(self.data_dir, train=True, download=True)
        RSKaggleDataset(self.data_dir, train=False, download=True)

        if self.google_maps_api != "":
            GoogleMapsDataset(self.data_dir, download=True)

    def setup(self, stage: Any = None) -> None:
        # Transform and split datasets
        kaggle_trainset = RSKaggleDataset(
            self.data_dir, train=True, transforms=self.transforms_train
        )
        if self.google_maps_api != "":
            google_maps_trainset = GoogleMapsDataset(
                self.data_dir, transforms=self.transforms_train, google_maps_api=self.google_maps_api
            )
            self.data_pretrain = google_maps_trainset

        testset = RSKaggleDataset(
            self.data_dir, train=False, transforms=self.transforms_test
        )

        

        self.data_train, self.data_val = random_split(
            kaggle_trainset, self.train_val_split
        )
        self.data_test = testset

    def train_dataloader(self) -> DataLoader:
        #Pretrain on google maps dataset and then train on kaggle dataset
        if self.google_maps_api != "":
            if self.trainer.current_epoch <= 60:
                if self.trainer.current_epoch <= 1:
                    print("Start training on google dataset")
                return DataLoader(
                    dataset=self.data_pretrain,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    shuffle=True,
                )
            else:
                if self.trainer.current_epoch <= 61:
                    print("Start training on kaggle dataset")
                return DataLoader(
                    dataset=self.data_train,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    shuffle=True,
                )
        #Train only on kaggle dataset if no google maps API key was given
        else:
            return DataLoader(
                dataset=self.data_train,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                shuffle=True,
            )

    def val_dataloader(self) -> Optional[DataLoader]:
        if self.train_val_split[1] == 0:
            return None

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
