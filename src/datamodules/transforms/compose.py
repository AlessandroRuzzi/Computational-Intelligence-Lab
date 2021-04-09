from typing import Any, Callable, List


class Compose:
    """
    Composes several transforms together.
    """

    def __init__(self, transforms: List[Callable]) -> None:
        self.transforms = transforms

    def __call__(self, *inputs: Any) -> Any:
        for transform in self.transforms:
            if not isinstance(inputs, (list, tuple)):
                inputs = [inputs]
            inputs = transform(*inputs)
        return inputs
