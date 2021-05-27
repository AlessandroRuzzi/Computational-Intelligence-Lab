from typing import List

import torch
from torch import nn, Tensor
from torchvision.models.resnet import ResNet
from torchvision.models.resnet import Bottleneck, BasicBlock

class ResNetEncoderBase(ResNet):
    
    def __init__(self, depth: int = 5, **kwargs) -> None:
        super().__init__(**kwargs)
        del self.fc
        del self.avgpool
        self.depth = depth
        self.blocks = [
            nn.Identity(),
            nn.Sequential(self.conv1, self.bn1, self.relu),
            nn.Sequential(self.maxpool, self.layer1),
            self.layer2,
            self.layer3,
            self.layer4,
        ]

    def forward(self, x: Tensor) -> List[Tensor]:
        outputs = []
        for i in range(self.depth + 1):
            x = self.blocks[i](x)
            outputs.append(x)
        return outputs


class ResNetEncoder(ResNetEncoderBase):

    def __init__(self, name: str, pretrained: bool = False, **kwargs) -> None:
        self.in_channels = self.models[name]["in_channels"] 
        self.out_channels = self.models[name]["out_channels"] 
        super().__init__(**self.models[name]["params"], **kwargs) 

        if pretrained:
            self.load_state_dict(torch.utils.model_zoo.load_url(self.models[name]["pretrained_weights"]))

    def load_state_dict(self, state_dict: str, **kwargs):
        state_dict.pop("fc.bias")
        state_dict.pop("fc.weight")
        super().load_state_dict(state_dict, **kwargs)

    models = {
        "resnet34": {
            "in_channels": 3,
            "out_channels": [3, 64, 64, 128, 256, 512],          
            "pretrained_weights": "https://download.pytorch.org/models/resnet34-b627a593.pth",
            "params": {
                "block": BasicBlock,
                "layers": [3, 4, 6, 3],
            },
        },
        "resnet50": {
            "in_channels": 3,
            "out_channels": [3, 64, 256, 512, 1024, 2048],            
            "pretrained_weights": "https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnet50-08389792.pth",
            "params": {
                "block": Bottleneck,
                "layers": [3, 4, 6, 3]
            }
        },
        "resnet101": {
            "in_channels": 3,
            "out_channels": [3, 64, 256, 512, 1024, 2048],            
            "pretrained_weights": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
            "params": {
                "block": Bottleneck,
                "layers": [3, 4, 23, 3]
            }
        },
        "resnext101_32x8d": {
            "in_channels": 3,
            "out_channels": [3, 64, 256, 512, 1024, 2048],
            "pretrained_weights": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
            "params": {
                "block": Bottleneck,
                "layers": [3, 4, 23, 3],
                "groups": 32,
                "width_per_group": 8,
            },
        },
        "resnext101_32x16d": {
            "in_channels": 3,
            "out_channels": [3, 64, 256, 512, 1024, 2048],
            "pretrained_weights": "https://download.pytorch.org/models/ig_resnext101_32x16-c6f796b0.pth",
            "params": {
                "block": Bottleneck,
                "layers": [3, 4, 23, 3],
                "groups": 32,
                "width_per_group": 16,
            },
        },
    }
