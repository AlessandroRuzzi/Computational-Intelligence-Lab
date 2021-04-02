from typing import Any

import torchvision.transforms.functional as F


class RandomFlip:
    """
    Flips image randomly.
    """

    def __init__(self, p: float) -> None:
        pass

    def __call__(self, *inputs: Any) -> Any:
        processed_images = []

        for index, _input in enumerate(inputs):
            processed_images.append(F.hflip(_input))

        return processed_images
