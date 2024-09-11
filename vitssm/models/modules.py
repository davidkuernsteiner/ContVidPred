from math import sqrt

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn
from xformers.components.activations import Activation
from xformers.components.attention import ScaledDotProduct
from xformers.components.feedforward import MLP
from xformers.components.multi_head_dispatch import MultiHeadDispatch


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, n_tokens: int, latent_dim: int, p_dropout: float = 0.0):
        super().__init__()

        self.pos_enc = nn.Parameter(torch.randn(1, n_tokens, latent_dim) * 0.02)
        self.dropout = nn.Dropout(p=p_dropout)

        self.n_tokens = n_tokens
        self.latent_dim = latent_dim

    def forward(self, x: Tensor):
        """Input dims: [batch, token, latent]
        Output dims: [batch, token, latent]
        """
        _, t, _ = x.shape
        if t != self.n_tokens:
            pos_enc = self.resample_pos_enc(t)
        else:
            pos_enc = self.pos_enc

        pos_enc = pos_enc.to(x.device)

        return self.dropout(x + pos_enc)

    def resample_pos_enc(self, new_n_tokens: int):
        new_hw = int(sqrt(new_n_tokens))
        new_size = new_hw, new_hw
        old_hw = int(sqrt(self.pos_enc.shape[1]))

        orig_dtype = self.pos_enc.dtype
        pos_enc = self.pos_enc.float()  # interpolate needs float32
        pos_enc = rearrange(pos_enc, "b (h w) e -> b e h w", h=old_hw, w=old_hw)
        pos_enc = F.interpolate(pos_enc, size=new_size, mode="bicubic", antialias=True)
        pos_enc = rearrange(pos_enc, "b e h w -> b (h w) e")
        pos_enc = pos_enc.to(orig_dtype)

        return pos_enc


class MixedCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        n_heads: int = 4,
        residual_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
        mlp_multiplier: int = 2,
    ) -> None:
        super().__init__()

        attention_mechanism = ScaledDotProduct()
        self.patch_attention = MultiHeadDispatch(
            dim_model=latent_dim,
            num_heads=n_heads,
            attention=attention_mechanism,
            residual_dropout=residual_dropout,
        )
        self.latent_attention = MultiHeadDispatch(
            dim_model=latent_dim,
            num_heads=n_heads,
            attention=attention_mechanism,
            residual_dropout=residual_dropout,
        )

        self.norm1 = nn.LayerNorm(latent_dim, eps=1e-6, elementwise_affine=True)
        self.norm2 = nn.LayerNorm(latent_dim, eps=1e-6, elementwise_affine=True)
        self.norm3 = nn.LayerNorm(latent_dim, eps=1e-6, elementwise_affine=True)
        self.norm4 = nn.LayerNorm(latent_dim, eps=1e-6, elementwise_affine=True)

        self.mlp1 = MLP(
            dim_model=latent_dim,
            dropout=mlp_dropout,
            activation=Activation.GeLU,
            hidden_layer_multiplier=mlp_multiplier,
        )
        self.mlp2 = MLP(
            dim_model=latent_dim,
            dropout=mlp_dropout,
            activation=Activation.GeLU,
            hidden_layer_multiplier=mlp_multiplier,
        )

    def forward(self, inputs: tuple[Tensor, Tensor, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        x_query, x_kv1, x_kv2 = inputs
        print(x_query.shape, x_kv1.shape, x_kv2.shape)
        x_query = x_query + self.norm1(self.patch_attention(x_query, key=x_kv1, value=x_kv1))
        x_query = x_query + self.norm2(self.mlp1(x_query))

        x_query = x_query + self.norm3(self.latent_attention(x_query, key=x_kv2, value=x_kv2))
        x_query = x_query + self.norm4(self.mlp2(x_query))

        return x_query, x_kv1, x_kv2
