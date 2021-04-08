from typing import List

import torch
import torch.nn as nn


class KaggleAccuracy(nn.Module):
    def __init__(self, patch_size: int = 16) -> None:
        super(KaggleAccuracy, self).__init__()
        self.patch_size = patch_size
        self.accuracy = 0

    def forward(self, preds : List[torch.Tensor],y : List[torch.Tensor]) -> float:
        self.accuracy = 0
        y_true = []
        y_preds = []
        for i in range(len(y)):
            y_true.append(
                y[i].reshape(
                    (
                        y[i].shape[0],
                        int(y[i].shape[1] / self.patch_size),
                        self.patch_size,
                        int(y[i].shape[2] / self.patch_size),
                        self.patch_size,
                    )
                )
            )
            y_preds.append(
                preds[i].reshape(
                    (
                        y[i].shape[0],
                        int(preds[i].shape[1] / self.patch_size),
                        self.patch_size,
                        int(preds[i].shape[2] / self.patch_size),
                        self.patch_size,
                    )
                )
            )
            y_true[i] = torch.where(
                torch.sum(y_true[i], dim=(2, 4)) / (self.patch_size ** 2) > 0.25, 1, 0
            )
            y_preds[i] = torch.where(
                torch.sum(y_preds[i], dim=(2, 4)) / (self.patch_size ** 2) > 0.25, 1, 0
            )
            print(y_preds[i])
            self.accuracy += torch.sum(y_true[i] == y_preds[i]) / torch.sum(
                torch.ones_like(y_true[i])
            )
        return self.accuracy / len(y)
