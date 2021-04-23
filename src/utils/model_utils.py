import os

import torch
import torchvision


def build_image_grid_train(
    x: torch.Tensor, y: torch.Tensor, preds: torch.Tensor, threshold: float = 0.5
) -> None:

    block = torch.cat(
        (
            x,
            y.expand(
                -1, 3, -1, -1
            ),  # Expand used to add 3 color channels to grayscale images
            preds.expand(-1, 3, -1, -1),
            (preds > threshold).expand(-1, 3, -1, -1),
        ),
        0,
    )

    img_grid = torchvision.utils.make_grid(block)
    img_grid = img_grid.permute((1, 2, 0))
    return img_grid


def build_image_grid_test(
    x: torch.Tensor, preds: torch.Tensor, threshold: float = 0.5
) -> None:

    block = torch.cat(
        (x, preds.expand(-1, 3, -1, -1), (preds > threshold).expand(-1, 3, -1, -1)), 0
    )

    img_grid = torchvision.utils.make_grid(block)
    img_grid = img_grid.permute((1, 2, 0))
    return img_grid


def save_images(path: str, ids: torch.Tensor, preds: torch.Tensor) -> None:
    os.makedirs(path, exist_ok=True)

    for i in range(preds.shape[0]):
        torchvision.utils.save_image(
            preds[i],
            os.path.join(
                path,
                f"satImage_{ids[i]:03.0f}.png",
            ),
        )
