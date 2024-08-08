from typing import Union, Tuple
from functools import reduce
from operator import mul
from einops import rearrange
from math import sqrt

import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor

from timm.layers.pos_embed import resample_abs_pos_embed



class LearnablePositionalEncoding(nn.Module):
    def __init__(
            self,
            n_tokens: int,
            latent_dim: int,
            p_dropout: float = 0.
        ):
        super().__init__()

        self.pos_enc = nn.Parameter(torch.randn(1, n_tokens, latent_dim) * .02)
        self.dropout = nn.Dropout(p=p_dropout)

        self.n_tokens = n_tokens
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor):
        """
        Input dims: [batch, token, latent]
        Output dims: [batch, token, latent]
        """
        _, t, _ = x.shape
        pos_enc = self.resample_pos_enc(t)

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
    