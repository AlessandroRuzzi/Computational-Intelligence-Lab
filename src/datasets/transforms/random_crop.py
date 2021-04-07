from typing import Any, Tuple

import torchvision.transforms.functional as F
from torchvision.transforms import transforms


class RandomCrop:
    """
    Crops a random part of the image.
    """

    def __init__(self, output_size: Tuple[int, int] = (200, 200)) -> None:
        self.output_size = output_size

    def __call__(self, *inputs: Any) -> Any:
        outputs = list(inputs)

        i, j, h, w = transforms.RandomCrop.get_params(inputs[0], self.output_size)

        for index, _input in enumerate(outputs):
            outputs[index] = F.crop(_input, i, j, h, w)

        return outputs
