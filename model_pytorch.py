import torch
import torch.nn as nn
import torch.nn.functional as F

class patchEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: int
    ):
        super().__init__()

        self.patch_size = patch_size
        self.patches = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        # x -> [B, C, H, W]
        x = self.patches(batch) # x -> [B, embed_dim, (H/patch_size), (W/patch_size)]
        x = x.flatten(2)        # x -> [B, embed_dim, Num Patches]
        x = x.transpose(1, 2)   # x -> [B, Num Patches, embed_dim]

        return x

class MultiheadSelfAttention(nn.Module):
    def __init__(
        self, 
        embed_dim: int,
         num_heads:int
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        self.scale = self.head_dim ** -0.5

        self.qkv_project = nn.Linear(
            in_features = embed_dim,
            out_features = embed_dim*3,
            bias=True
        )

        self.out_project = nn.Linear(
            in_features = embed_dim,
            out_features = embed_dim
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = x.shape 
        
        # [B, Num Patches, embed_dim]
        qkv = self.qkv_project(x).reshape(batch, seq_len, 3, self.num_heads, self.head_dim)
        # [B, Num Patches, 3, n_h, h_d]
        qkv = qkv.permute(2, 0, 3, 1, 4)
        # [3, B, n_h, Num Patches, h_d]

        q, k, v = qkv[0], qkv[1], qkv[2] # [B, n_h, Num Patches, h_d]

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_probs = F.softmax(attn_scores, dim=-1)

        attn_out = torch.matmul(attn_probs, v).transpose(1, 2) # [B, Num_patches, n_h, h_d]
        attn_out = attn_out.reshape(batch, seq_len, dim) # [B, Num Patches, embed_dim]

        out = self.out_project(attn_out) # [B, Num Patches, embed_dim]
 
        return out

class MLP(nn.Module):
    def __init__(
        self, 
        in_features: int, 
        hidden_features: int, 
        out_features: int, 
        dropout: float
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU() # ViT uses GELU
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int, 
        mlp_ratio: int, 
        dropout: float
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.multiheadattn = MultiheadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(
            in_features=embed_dim,
            out_features=embed_dim * mlp_ratio,
            dropout=dropout
        )


    def forward(self, x: torch.Tensor):
        # Attention block
        attention_out = self.multiheadattn(self.norm1(x))
        x = x + attention_out

        # Feedforward block
        feedforaward_out = self.mlp(self.norm2(x))
        x = x + feedforaward_out

        return x

