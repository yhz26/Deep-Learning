import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# 残差连接，应用在每个前馈网络和注意力层之后
class SkipConnection(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x, **kwargs):
        return self.layer(x, **kwargs) + x


# LayerNorm归一化，应用在多头注意力和激活函数层
class NormalizedLayer(nn.Module):
    def __init__(self, dim, layer):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.layer = layer

    def forward(self, x, **kwargs):
        return self.layer(self.layer_norm(x), **kwargs)


# 前馈神经网络，包含两个线性层和一个激活函数
class MLPBlock(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.mlp(x)


# 多头自注意力层
class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.scale_factor = dim ** -0.5

        self.qkv_projection = nn.Linear(dim, dim * 3, bias=False)
        self.output_projection = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.num_heads
        qkv = self.qkv_projection(x)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=h)

        attention_scores = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale_factor

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == attention_scores.shape[-1], 'Mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            attention_scores.masked_fill_(~mask, float('-inf'))

        attention_weights = attention_scores.softmax(dim=-1)

        output = torch.einsum('bhij,bhjd->bhid', attention_weights, v)
        output = rearrange(output, 'b h n d -> b n (h d)')
        return self.output_projection(output)


# Transformer模块，包含多个自注意力和前馈层
class TransformerBlock(nn.Module):
    def __init__(self, dim, depth, num_heads, mlp_dim):
        super().__init__()
        self.attention_layers = nn.ModuleList([
                                                  SkipConnection(
                                                      NormalizedLayer(dim, MultiHeadAttention(dim, num_heads)))
                                              ] * depth)

        self.feedforward_layers = nn.ModuleList([
                                                    SkipConnection(NormalizedLayer(dim, MLPBlock(dim, mlp_dim)))
                                                ] * depth)

    def forward(self, x, mask=None):
        for attn_layer, ff_layer in zip(self.attention_layers, self.feedforward_layers):
            x = attn_layer(x, mask=mask)
            x = ff_layer(x)
        return x


# 图像分类的ViT模型，基于Transformer结构
class ViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, num_heads, mlp_dim, channels=3):
        super().__init__()
        assert image_size % patch_size == 0, 'Image size must be divisible by patch size'

        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size

        self.position_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = TransformerBlock(dim, depth, num_heads, mlp_dim)

        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, x, mask=None):
        p = self.patch_size
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        x = self.patch_to_embedding(x)

        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.position_embedding
        x = self.transformer(x, mask)

        x = x[:, 0]
        return self.mlp_head(x)
