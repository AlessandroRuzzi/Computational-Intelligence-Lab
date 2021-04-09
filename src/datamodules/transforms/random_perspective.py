from typing import Any, List

import torchvision.transforms.functional as F
from torchvision.transforms import transforms


class RandomPerspective:
    """
    Applies a random perspective transformation.
    """

    def __init__(
        self, img_size: List[int] = [400, 400], distortion_scale: float = 0.5
    ) -> None:
        self.img_size = img_size
        self.distortion_scale = distortion_scale

    def __call__(self, *inputs: Any) -> Any:
        outputs = list(inputs)

        points_start, points_end = transforms.RandomPerspective.get_params(
            width=self.img_size[0],
            height=self.img_size[1],
            distortion_scale=self.distortion_scale,
        )

        for index, _input in enumerate(outputs):
            outputs[index] = F.perspective(_input, points_start, points_end)

        return outputs
