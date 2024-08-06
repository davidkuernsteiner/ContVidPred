from typing import Union, Tuple


from einops import rearrange
from torch import nn
from torch import Tensor
from timm.models.vision_transformer import VisionTransformer
from xformers.components.positional_embedding import SinePositionalEmbedding
from xformers.components.attention import ScaledDotProduct
from xformers.components.feedforward import MLP



class CausalTransformerBLock(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 768,
        seq_len: int = 256,
        to_seq_len: int = 256
    ) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(embedding_dim, eps=1e-6, elementwise_affine=True)
        self.norm2 = nn.LayerNorm(embedding_dim, eps=1e-6, elementwise_affine=True)
        self.norm3 = nn.LayerNorm(embedding_dim, eps=1e-6, elementwise_affine=True)

        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)

        self.attn = ScaledDotProduct(dropout=0.1, causal=True)
        self.mlp = MLP(embedding_dim, embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x += self.attn(x, x, x)

        x = self.ln1(x)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        x = self.attn(q, k, v)
        x = self.out_proj(x)
        x = self.ln2
        out = self.norm(out)

        return out