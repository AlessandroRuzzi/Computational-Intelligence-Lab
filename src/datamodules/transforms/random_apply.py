from typing import Any, Callable, List

import torch


class RandomApply:
    """
    Applies a random affine transformation.
    """

    def __init__(self, transforms: List[Callable], p: float = 0.5) -> None:
        self.transforms = transforms
        self.p = p

    def __call__(self, *inputs: Any) -> Any:
        outputs = list(inputs)

        for transform in self.transforms:
            if torch.rand(1) > self.p:
                outputs = transform(*inputs)
        return outputs


"""
    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) > self.p:
            for transform in self.transforms:
                    inputs = transform(*inputs)
        return inputs
"""
