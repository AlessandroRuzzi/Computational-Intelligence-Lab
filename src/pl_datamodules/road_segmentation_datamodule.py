from typing import Any, Callable

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from src.datasets.road_segmentation_dataset import RoadSegmentationDataset
from src.datasets.transforms.to_tensor import ToTensor


class RoadSegmentationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 64,
        transforms: Callable = ToTensor(),
        train_val_split: tuple = (80, 20),
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

        self.transforms_train = transforms
        self.transforms_test = ToTensor()

        self.dims = (3, 400, 400)

        self.data_train: Any = None
        self.data_val: Any = None
        self.data_test: Any = None

    def get_transforms(self) -> Any:
        return self.transforms_train

    def prepare_data(self) -> None:
        # Download data
        RoadSegmentationDataset(self.data_dir, train=True, download=True)
        RoadSegmentationDataset(self.data_dir, train=False, download=True)

    def setup(self, stage: Any = None) -> None:
        # Transform and split datasets
        trainset = RoadSegmentationDataset(
            self.data_dir, train=True, transforms=self.transforms_train
        )
        testset = RoadSegmentationDataset(
            self.data_dir, train=False, transforms=self.transforms_test
        )
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
