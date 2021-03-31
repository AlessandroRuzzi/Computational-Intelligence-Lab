import torch
from torch import nn


class SimpleNet(nn.Module):
    def __init__(self, hparams: dict) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(hparams["input_size"], hparams["hidden_size"]),
            nn.ReLU(),
            nn.Linear(hparams["hidden_size"], hparams["output_size"]),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, width, height = x.size()
        # (batch_size, 1, width, height) -> (batch_size, 1*width*height)
        x = x.view(batch_size, -1)
        x = self.model(x)
        return x
