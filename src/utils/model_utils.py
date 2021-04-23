# flake8: noqa
# mypy: ignore-errors

import os
import re
from typing import Generator

import matplotlib.image as mpimg
import numpy as np
import torch
import torchvision


def gray_to_rgb(x: torch.Tensor) -> torch.Tensor:
    return x.expand(-1, 3, -1, -1)


def build_image_grid_train(
    x: torch.Tensor, y: torch.Tensor, preds: torch.Tensor, threshold: float = 0.5
) -> None:

    block = torch.cat(
        (
            x,
            gray_to_rgb(y),
            gray_to_rgb(preds),
            gray_to_rgb((preds > threshold)),
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
        (
            x,
            gray_to_rgb(preds),
            gray_to_rgb((preds > threshold)),
        ),
        0,
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


def mask_to_submission_strings(
    path: str, image_filename: str, patch_size: int, threshold: float = 0.25
) -> Generator:

    img_number = int(re.search(r"\d+", image_filename).group(0))
    im = mpimg.imread(os.path.join(path, image_filename))
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i : i + patch_size, j : j + patch_size]
            label = int(np.mean(patch) > threshold)
            yield (f"{img_number:03d}_{j}_{i},{label}")


def images_to_csv(path: str, csv_filename: str, patch_size: int = 16) -> str:

    output_file = os.path.join(path, csv_filename)

    # Read all files from given path
    image_filenames = [
        filename
        for filename in os.listdir(path)
        if os.path.isfile(os.path.join(path, filename))
    ]

    # Converts images into a submission file
    with open(output_file, "w") as f:
        f.write("id,prediction\n")
        for filename in image_filenames:
            f.writelines(
                f"{s}\n" for s in mask_to_submission_strings(path, filename, patch_size)
            )

    return output_file
