from typing import Any

import torch
import torchvision.transforms.functional as F


class RandomFlip:
    """
    Flips image randomly.
    """

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, *inputs: Any) -> Any:
        processed_images = []
        if torch.rand(1) > self.p:
            for index, _input in enumerate(inputs):
                processed_images.append(F.hflip(_input))
        else:
            for index, _input in enumerate(inputs):
                processed_images.append(F.vflip(_input))

        return processed_images
