from typing import Any, List

import torchvision.transforms.functional as F
from torchvision.transforms import transforms


class RandomAffine:
    """
    Applies a random affine transformation.
    """

    def __init__(
        self,
        img_size: List[int] = None,
        degrees: List[float] = None,
        translate: List[float] = None,
        scale_ranges: List[float] = None,
        shears: List[float] = None,
    ) -> None:
        if img_size is None:
            img_size = [400, 400]
        if degrees is None:
            degrees = [90, 90]
        self.img_size = img_size
        self.degrees = degrees
        self.translate = translate
        self.scale_ranges = scale_ranges
        self.shears = shears

    def __call__(self, *inputs: Any) -> Any:
        outputs = list(inputs)

        i, j, w, h, = transforms.RandomAffine.get_params(
            self.degrees,
            translate=None,
            scale_ranges=None,
            shears=None,
            img_size=self.img_size,
        )

        for index, _input in enumerate(inputs):
            outputs[index] = F.affine(_input, i, list(j), w, list(h))

        return outputs
