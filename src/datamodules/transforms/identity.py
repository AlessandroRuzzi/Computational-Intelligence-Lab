from typing import Any


class Identity:
    """
    Doesn't apply any transform, useful e.g. in random choice to keep the original image sometimes
    """

    def __init__(self) -> None:
        pass

    def __call__(self, *inputs: Any) -> Any:
        return inputs
