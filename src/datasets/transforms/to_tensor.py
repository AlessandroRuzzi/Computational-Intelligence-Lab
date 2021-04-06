from typing import Any

from torchvision import transforms


class ToTensor:
    """
    Converts all inputs to tensors.
    """

    def __init__(self) -> None:
        pass

    def __call__(self, *inputs: Any) -> Any:
        outputs = list(inputs)

        transform_to_tensor = transforms.ToTensor()

        for index, _input in enumerate(inputs):
            outputs[index] = transform_to_tensor(_input)

        return outputs
