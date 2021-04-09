from typing import Any

from torchvision import transforms


class ToTensor:
    """
    Converts all inputs to tensors.
    """

    def __init__(self) -> None:
        self.transform = transforms.ToTensor()

    def __call__(self, *inputs: Any) -> Any:
        outputs = list(inputs)

        for index, _input in enumerate(outputs):
            outputs[index] = self.transform(_input)

        return outputs
