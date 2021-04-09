from typing import Any, Callable, List


class Distinct:
    """
    Applies distinct transforms to each input (e.g. first transform applied on first input, second transfrom on second input and so on...)
    """

    def __init__(self, transforms: List[Callable]) -> None:
        self.transforms = transforms

    def __call__(self, *inputs: Any) -> Any:
        outputs = list(inputs)

        for index, transform in enumerate(self.transforms):
            outputs[index] = transform(inputs[index])

        return outputs
