import torch
import torch.nn as nn
import torch.nn.functional as F

class patchEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        dim: int = 768,
        patch_size: int = 16
        ):
        super().__init__()

        self.patch_size = patch_size
        self.patches = nn.Conv2d(
            in_channels=in_channels,
            out_channels=dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        # x -> [B, C, H, W]
        x = self.patches(batch) # x -> [B, dim, (H/p_s), (W/p_s)]
        x = x.flatten(2)        # x -> [B, dim, Num Patches]
        x = x.transpose(1, 2)   # x -> [B, Num Patches, dim]

        return x

class MultiheadSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads:int):
        super().__init__()
        self.dim = dim
        self.num_heads = heads

        self.head_dim = dim // num_heads

        

        

