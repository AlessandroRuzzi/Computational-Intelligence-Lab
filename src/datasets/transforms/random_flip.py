import random
from typing import Any

import torchvision.transforms.functional as F


class RandomFlip:
    """
    Flips image randomly.
    """

    def __init__(self) -> None:
        pass

    def __call__(self, *inputs: Any) -> Any:
        processed_images = []
        if random.random() > 0.5:
            for index, _input in enumerate(inputs):
                processed_images.append(F.hflip(_input))
        else:
            for index, _input in enumerate(inputs):
                processed_images.append(F.vflip(_input))

        return processed_images
