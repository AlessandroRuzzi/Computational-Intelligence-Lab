import random
from typing import Any, Callable, List


class RandomChoiceCompose:
    """
    Randomly choose to apply one transform from a collection of transforms.
    """

    def __init__(self, transforms: List[Callable]) -> None:
        self.transforms = transforms

    def __call__(self, *inputs: Any) -> Any:
        transform = random.choice(self.transforms)
        outputs = transform(*inputs)
        return outputs
