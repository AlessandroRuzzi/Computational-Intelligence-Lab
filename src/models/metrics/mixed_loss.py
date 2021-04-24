from typing import List

import torch
import torch.nn as nn


class MixedLoss(nn.Module):
    def __init__(self, weights: List[float], losses: List[nn.Module]):
        super().__init__()
        self.weights = weights
        self.losses = losses
        assert len(weights) == len(losses), "weights len doesn't match losses len"
        assert sum(weights) == 1.0, "weights should sum to 1"

    def forward(self, predict: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        loss = torch.tensor(0)

        for i in range(len(self.weights)):
            loss += self.weights[i] * self.losses[i](predict, target)

        return loss
