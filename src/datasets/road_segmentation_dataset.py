import os
import re
import zipfile
from typing import Any, Callable, List, Optional, Tuple

import torch
from PIL import Image


class RoadSegmentationDataset(torch.utils.data.Dataset):

    kaggle_competition = "cil-road-segmentation-2021"
    kaggle_folder_train_images = "training/training/images/"
    kaggle_folder_train_masks = "training/training/groundtruth/"
    kaggle_folder_test_images = "test_images/test_images/"

    @property
    def folder_raw(self) -> str:
        return os.path.join(
            self.root, self._camel_to_snake(self.__class__.__name__), "raw"
        )

    @property
    def folder_processed(self) -> str:
        return os.path.join(
            self.root, self._camel_to_snake(self.__class__.__name__), "processed"
        )

    @property
    def folder_train(self) -> str:
        return os.path.join(self.folder_processed, "train")

    @property
    def folder_test(self) -> str:
        return os.path.join(self.folder_processed, "test")

    def __init__(
        self,
        root: str = "data/",
        train: bool = True,
        download: bool = False,
        transforms: Optional[Callable] = None,
    ) -> None:
        self.root = root
        self.train = train
        self.transforms = transforms
        self.images: List[str] = []
        self.masks: List[str] = []

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found." + " You can use download=True to download it"
            )

        self.process()

    def download(self) -> None:

        if self._check_exists():
            return

        os.makedirs(self.folder_raw, exist_ok=True)
        os.makedirs(self.folder_processed, exist_ok=True)

        from kaggle import api

        api.authenticate()

        print(f"Downloading competition dataset <{self.kaggle_competition}>.")
        api.competition_download_files(self.kaggle_competition, path=self.folder_raw)

        self.unzip()

    def unzip(self) -> None:
        print("Unzipping.")
        source_zip = os.path.join(self.folder_raw, self.kaggle_competition) + ".zip"
        zipdata = zipfile.ZipFile(source_zip)
        # Extract either train or test images and rename them.
        for zipinfo in zipdata.infolist():
            img_name = zipinfo.filename
            if self.train:
                if img_name.startswith(self.kaggle_folder_train_images):
                    zipinfo.filename = (
                        f"{self._img_number_from_name(img_name)-1:03d}_image.png"
                    )
                    zipdata.extract(zipinfo, self.folder_train)
                    self.images.append(zipinfo.filename)
                if img_name.startswith(self.kaggle_folder_train_masks):
                    zipinfo.filename = (
                        f"{self._img_number_from_name(img_name)-1:03d}_mask.png"
                    )
                    zipdata.extract(zipinfo, self.folder_train)
                    self.masks.append(zipinfo.filename)
            else:
                if img_name.startswith(self.kaggle_folder_test_images):
                    zipinfo.filename = (
                        f"{self._img_number_from_name(img_name)-1:03d}_image.png"
                    )
                    zipdata.extract(zipinfo, self.folder_test)
                    self.images.append(zipinfo.filename)

    def process(self) -> None:

        images, masks = [], []

        if self.train:
            folder_train_files = [
                f
                for f in os.listdir(self.folder_train)
                if os.path.isfile(os.path.join(self.folder_train, f))
            ]
            images = [
                os.path.join(self.folder_train, f)
                for f in folder_train_files
                if "image" in f
            ]
            masks = [
                os.path.join(self.folder_train, f)
                for f in folder_train_files
                if "mask" in f
            ]
        else:
            folder_test_files = [
                f
                for f in os.listdir(self.folder_train)
                if os.path.isfile(os.path.join(self.folder_train, f))
            ]
            images = [
                os.path.join(self.folder_test, f)
                for f in folder_test_files
                if "image" in f
            ]

        self.images = sorted(images)
        self.masks = sorted(masks)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        image = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.masks[index])

        if self.transforms is not None:
            image, mask = self.transforms(image, mask)

        return image, mask

    def __len__(self) -> int:
        return len(self.images)

    def _check_exists(self) -> bool:
        print(self.folder_train)
        if self.train:
            return os.path.exists(self.folder_train)
        else:
            return os.path.exists(self.folder_test)

    def _camel_to_snake(self, name: str) -> str:
        name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
        return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()

    def _img_number_from_name(self, name: str) -> int:
        return int(re.compile(r"\d+").findall(name)[0])
