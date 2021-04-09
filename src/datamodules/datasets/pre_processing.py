"""
import torch
import torchvision
from torchvision import transforms
from torchvision.datasets.vision import StandardTransform

from src.datasets.road_segmentation_dataset import RoadSegmentationDataset
from src.datasets.transforms.random_affine import RandomAffine
from src.datasets.transforms.random_apply import RandomApply
from src.datasets.transforms.random_crop import RandomCrop
from src.datasets.transforms.random_flip import RandomFlip
from src.utils.template_utils import imshow


class PreProcessor:
    def __init__(self, root: str = "data/") -> None:
        self.root = root
        self.pre_process_erasing()

    def pre_process_erasing(self) -> None:
        transform_image = torchvision.transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        transform_mask = transforms.Compose([transforms.ToTensor()])

        standard_tranform = StandardTransform(transform_image, transform_mask)
        dataset = RoadSegmentationDataset(
            self.root, train=True, download=True, transforms=standard_tranform
        )
        image, mask = dataset.__getitem__(40)

        imshow(image)
        imshow(mask)

        # flake8: noqa
        cropper = RandomCrop()
        # flake8: noqa
        flip = RandomFlip()
        affine = RandomAffine(degrees=[30, 30])
        apply = RandomApply(transforms=[flip, affine, cropper])
        images = apply(image, mask)
        for image in images:
            imshow(image)

        print(image.size(), mask.size(), image.dtype, mask.dtype)


def main() -> None:
    PreProcessor()


if __name__ == "__main__":
    main()
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
from torchvision.datasets.vision import StandardTransform

from src.datasets.road_segmentation_dataset import RoadSegmentationDataset
from src.utils.template_utils import imshow

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device is {device}.")


class PreProcessor:
    def __init__(self, root: str = "data/") -> None:
        self.root = root
        self.pre_process_erasing()

    def pre_process_erasing(self) -> None:
        transform_image = torchvision.transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        transform_mask = transforms.Compose([transforms.ToTensor()])

        standard_tranform = StandardTransform(transform_image, transform_mask)
        dataset = RoadSegmentationDataset(
            self.root, train=True, download=True, transforms=standard_tranform
        )
        image, mask = dataset.__getitem__(40)

        imshow(image)
        imshow(mask)
        mask = mask.reshape(-1)
        mask = np.asarray(mask)

        print(mask.shape)
        for i in range(len(mask)):
            if mask[i] > 0.5:
                mask[i] = 1
            else:
                mask[i] = 0
        mask = torch.tensor(mask)
        mask = mask.reshape(1, 400, 400)

        imshow(mask)

        kernel = np.ones((3, 3), np.uint8)
        mask = np.asarray(mask)
        _, mask = cv2.threshold(mask, 0.9, 1, cv2.THRESH_BINARY)
        mask = cv2.erode(mask, kernel, iterations=7)
        mask = cv2.dilate(mask, kernel, iterations=7)
        mask = torch.tensor(mask)
        imshow(mask)

        print(image.size(), mask.size(), image.dtype, mask.dtype)


def main() -> None:
    PreProcessor()


if __name__ == "__main__":
    main()


# find what type of preprocessing we can do
# see how they flatten images in the notebooks
#
