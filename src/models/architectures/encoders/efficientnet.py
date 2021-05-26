from typing import Any, Dict, List

import torch
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import get_model_params
from torch import Tensor, nn


class EfficientNetEncoder(EfficientNet):
    def __init__(
        self, name: str, depth: int = 5, pretrained: bool = True, dilated: bool = False
    ) -> None:
        blocks_args, global_params = get_model_params(name, override_params=None)
        super().__init__(blocks_args, global_params)
        args = self.models[name]
        self.stage_idxs = args["params"]["stage_idxs"]
        self.out_channels = args["out_channels"]
        self.in_channels = args["in_channels"]
        self.depth = depth
        del self._fc

        self.blocks = [
            nn.Identity(),
            nn.Sequential(self._conv_stem, self._bn0, self._swish),
            self._blocks[: self.stage_idxs[0]],
            self._blocks[self.stage_idxs[0] : self.stage_idxs[1]],
            self._blocks[self.stage_idxs[1] : self.stage_idxs[2]],
            self._blocks[self.stage_idxs[2] :]
        ]

        if pretrained:
            self.load_state_dict(
                torch.utils.model_zoo.load_url(args["pretrained_weights"])
            )

        if dilated:
            self.make_last_dilated()

    def forward(self, x: Tensor) -> List[Tensor]:
        block_number = 0.0
        drop_connect_rate = self._global_params.drop_connect_rate
        features = []

        for i in range(self.depth + 1):
            if i < 2:
                x = self.blocks[i](x)
            else:
                for module in self.blocks[i]:
                    drop_connect = drop_connect_rate * block_number / len(self._blocks)
                    block_number += 1.0
                    x = module(x, drop_connect)
            features.append(x)

        return features

    def load_state_dict(self, state_dict: Dict[str, Tensor], **kwargs: Any) -> None:
        state_dict.pop("_fc.bias")
        state_dict.pop("_fc.weight")
        super().load_state_dict(state_dict, **kwargs)

    def make_last_dilated(self, dilation_rate: int = 2) -> None:
        """ Useful for DeepLabV3Plus decoder. """
        last_idx = self.depth
        for mod in self.blocks[last_idx].modules():
            if isinstance(mod, nn.Conv2d):
                mod.stride = (1, 1)
                mod.dilation = (dilation_rate, dilation_rate)
                kh, kw = mod.kernel_size
                mod.padding = ((kh // 2) * dilation_rate, (kh // 2) * dilation_rate)
                if hasattr(mod, "static_padding"):
                    mod.static_padding = nn.Identity()

    models = {
        "efficientnet-b0": {
            "in_channels": 3,
            "out_channels": [3, 32, 24, 40, 112, 320],
            "pretrained_weights": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth",
            "params": {
                "stage_idxs": (3, 5, 9, 16),
            },
        },
        "efficientnet-b1": {
            "in_channels": 3,
            "out_channels": [3, 32, 24, 40, 112, 320],
            "pretrained_weights": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b1-f1951068.pth",
            "params": {
                "stage_idxs": (5, 8, 16, 23),
            },
        },
        "efficientnet-b2": {
            "in_channels": 3,
            "out_channels": [3, 32, 24, 48, 120, 352],
            "pretrained_weights": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b2-8bb594d6.pth",
            "params": {
                "stage_idxs": (5, 8, 16, 23),
            },
        },
        "efficientnet-b3": {
            "in_channels": 3,
            "out_channels": [3, 40, 32, 48, 136, 384],
            "pretrained_weights": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b3-5fb5a3c3.pth",
            "params": {
                "stage_idxs": (5, 8, 18, 26),
            },
        },
        "efficientnet-b4": {
            "in_channels": 3,
            "out_channels": [3, 48, 32, 56, 160, 448],
            "pretrained_weights": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b4-6ed6700e.pth",
            "params": {
                "stage_idxs": (6, 10, 22, 32),
            },
        },
        "efficientnet-b5": {
            "in_channels": 3,
            "out_channels": [3, 48, 40, 64, 176, 512],
            "pretrained_weights": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b5-b6417697.pth",
            "params": {
                "stage_idxs": (8, 13, 27, 39),
            },
        },
        "efficientnet-b6": {
            "in_channels": 3,
            "out_channels": [3, 56, 40, 72, 200, 576],
            "pretrained_weights": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b6-c76e70fd.pth",
            "params": {
                "stage_idxs": (9, 15, 31, 45),
            },
        },
        "efficientnet-b7": {
            "in_channels": 3,
            "out_channels": [3, 64, 48, 80, 224, 640],
            "pretrained_weights": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b7-dcc49843.pth",
            "params": {
                "stage_idxs": (11, 18, 38, 55),
            },
        },
    }
