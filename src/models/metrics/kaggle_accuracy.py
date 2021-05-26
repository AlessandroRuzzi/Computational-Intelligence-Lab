import torch
from torch import nn, Tensor 

class KaggleAccuracy(nn.Module):
    
    def __init__(
        self, 
        threshold: float = 0.25,
        num_patches: int = 38, 
        size: int = 418
    ) -> None:
        super().__init__()  
        self.threshold = threshold
        self.num_patches = num_patches
        self.patch_size = size // num_patches 
        self.resize = nn.Upsample(size = size)
        self.unfold = nn.Unfold(
            kernel_size = self.patch_size,
            stride = self.patch_size
        )
    
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        x = self.to_patch_value(x)
        y = self.to_patch_value(y)
        acc = torch.mean((x == y).float()) 
        return acc
    
    def to_patch_value(self, x: Tensor) -> Tensor:
        x = x.float()
        x = self.resize(x)
        x = self.unfold(x)
        x = x.mean(dim = 1)
        return (x > self.threshold).float()