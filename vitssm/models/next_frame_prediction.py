from typing import Union, Tuple


from einops import rearrange
import torch
from torch import nn
from torch import Tensor
from timm.models.vision_transformer import VisionTransformer
from xformers.components.positional_embedding import SinePositionalEmbedding
from xformers.components.attention import ScaledDotProduct, LocalAttention
from xformers.components.multi_head_dispatch import MultiHeadDispatch

from .modules import LearnablePositionalEncoding


class LatentNextFramePrediction(nn.Module):
    def __init__(
        self,
        frame_in_size: Tuple[int, int] = (224, 224),
        frame_out_size: Tuple[int, int] = (224, 224),
        n_frames: int = 30,
        window_size: int = 10,
        patch_size: int = 16,
        latent_dim: int = 768,
        n_heads: int = 8,
        pos_enc_dropout: float = 0.
    ):
        super().__init__()
        self.frame_encoder = FrameEncoder(
            frame_in_size=frame_in_size,
            patch_size=patch_size,
            latent_dim=latent_dim,
        )
        self.latent_predictor = LatentPredictor(
            latent_dim=latent_dim,
            n_heads=n_heads,
            n_blocks=4,
            n_frames=n_frames,
            window_size=window_size,
        )
        self.latent_frame_decoder = LatentFrameDecoder()

        self.patchify = nn.Conv2d(
            3, 
            latent_dim, 
            kernel_size=patch_size, 
            stride=patch_size,
            )
        self.pos_enc_patches = LearnablePositionalEncoding(
            n_tokens=(frame_in_size // patch_size)**2, 
            latent_dim=latent_dim, 
            p_dropout=pos_enc_dropout,
        )

        self.frame_in_size = frame_in_size
        self.frame_out_size = frame_out_size
        self.n_frames = n_frames
        self.window_size = window_size
        self.patch_size = patch_size
        self.latent_dim = latent_dim

    def forward(self, x: Tensor):
        """
        Input dims: [batch, time, height, width, channel]
        Output dims: [batch, time, height, width, channel]
        """
        b, t, h, w, c = x.shape
        hp, wp = h // self.patch_size, w // self.patch_size

        x_patches = rearrange(x, "b t h w c -> (b t) c h w")
        x_patches = self.patchify(x_patches)
        x_patches = rearrange(x_patches, "(b t) e hp wp -> (b t) (hp wp) e", b=b, t=t)
        x_patches = self.pos_enc_patches(x_patches)
        x_patches = rearrange(x_patches, "(b t) (hp wp) e -> b t (hp wp) e", b=b, t=t, wp=wp, hp=hp)

        x_latent = rearrange(x, "b t h w c -> (b t) c h w")
        x_latent = self.frame_encoder(x_latent)
        x_latent = rearrange(x_latent, "(b t) e -> b t e", b=b, t=t)
        #x_latent = self.pos_enc(x_latent)
        x_next_latent = self.latent_predictor(x_latent)

        canvas = self.pos_enc_patches(torch.ones(b * t, (self.frame_out_size // self.patch_size)**2, self.latent_dim))
        x_next_frame = self.latent_frame_decoder(x_next_latent, x_patches, canvas)

        return x_next_frame


class FrameEncoder(nn.Module):
    def __init__(
        self,
        frame_in_size: Union[int, Tuple[int, int]] = 256,
        patch_size: int = 16,
        depth: int = 8,
        latent_dim: int = 96,
        n_heads: int = 8,
        mlp_ratio: float = 3.0,
        qkv_bias: bool = False,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()
        self.backbone = VisionTransformer(
            img_size=frame_in_size,
            patch_size=patch_size,
            embed_dim=latent_dim,
            depth=depth,
            num_heads=n_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            num_classes=0,
            class_token=True,
        )

    def forward(self, x: Tensor):
        """
        Input dims: [batch * time, channel, height, width]
        Output dims: [batch * time, latent]
        """
        return self.backbone(x)


class LatentPredictor(nn.Module):
    def __init__(
        self,
        latent_dim: int = 768,
        n_heads: int = 8,
        n_blocks: int = 4,
        n_frames: int = 302,
        window_size: int = 10,
    ) -> None:
        super().__init__()
        attention_mechanism = LocalAttention(causal=True, window_size=window_size)

        self.pos_enc = LearnablePositionalEncoding(n_tokens=n_frames, latent_dim=latent_dim)
        self.blocks = nn.Sequential(
            *(MultiHeadDispatch(latent_dim, n_heads, attention_mechanism) for _ in range(n_blocks))
        )

    def forward(self, x: torch.Tensor):
        """
        Input dims: [batch, time, latent]
        Output dims: [batch, time, latent]
        """
        return self.blocks(self.pos_enc(x))


class LatentFrameDecoder(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(
            self,
            x_latent: torch.Tensor,
            x_patches: torch.Tensor, 
            canvas: torch.Tensor,
        ):
        """
        Input dims:
            latent predictor: [batch, time, latent]
            frame patchify: [batch, time, height_patches, width_patches, latent]

        Output dims: [batch, time, height_out, width_out, channel]
        """
        print(f"x_latent: {x_latent.shape}\nx_patches: {x_patches.shape}\ncanvas: {canvas.shape}")
