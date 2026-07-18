import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
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

class TransformerBlock(nn.Module):
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

class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        in_channels: int = 3,
        patch_size: int = 16,
        embed_dim: int = 768,
        num_heads: int = 8,
        depth: int = 12,
        mlp_ratio: int = 4,
        mlp_droput: float = 0.5,
        num_classes: int = 1000

    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size
        )
        num_patches = (img_size // patch_size) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.blocks = nn.ModuleList(*
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=mlp_droput
                ) for _ in range(depth)
            ]
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(
                in_features = embed_dim, 
                out_features = num_classes)
        )

        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]

        # 1. Image patching
        x = self.patch_embed(x)

        # 2. Adding CLS token
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        # 3. Adding Positional embeddings
        x = x + self.pos_embedding

        # 4. Encoder blocks pass
        x = self.blocks(x)
        
        cls_token_out = x[:, 0]

        # 5. MLP head pass
        logits = self.mlp_head(cls_token_out)

        return logits
