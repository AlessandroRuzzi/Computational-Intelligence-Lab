from typing import Any

import torchvision.transforms.functional as F
from torchvision.transforms import transforms


class RandomCrop:
    """
    Crops a random part of the image.
    """

    def __init__(self, output_size: tuple) -> None:
        self.output_size = output_size

    def __call__(self, *inputs: Any) -> Any:
        processed_images = []
        i, j, h, w = transforms.RandomCrop.get_params(inputs[0], self.output_size)

        for index, _input in enumerate(inputs):
            processed_images.append(F.crop(_input, i, j, h, w))

        return processed_images
