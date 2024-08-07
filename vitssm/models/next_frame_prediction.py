from typing import Union, Tuple


from einops import rearrange
import torch
from torch import nn
from torch import Tensor
from timm.models.vision_transformer import VisionTransformer
from xformers.components.positional_embedding import SinePositionalEmbedding
from xformers.components.attention import ScaledDotProduct, LocalAttention
from xformers.components.multi_head_dispatch import MultiHeadDispatch


class LatentNextFramePrediction(nn.Module):
    def __init__(
        self,
        frame_in_size: Union[int, Tuple[int, int]] = 224,
        frame_out_size: Union[int, Tuple[int, int]] = 224,
        n_frames: int = 10,
        patch_size: int = 16,
        latent_dim: int = 768,
        num_heads: int = 8,
    ):
        super().__init__()
        self.frame_encoder = FrameEncoder(
            frame_in_size=frame_in_size,
            patch_size=patch_size,
            latent_dim=latent_dim,
        )
        self.latent_predictor = LatentPredictor(
            latent_dim=latent_dim,
            num_heads=num_heads,
            window_size=n_frames,
        )
        self.latent_frame_decoder = LatentFrameDecoder()

        self.patchify = nn.Conv2d(3, latent_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_enc = nn.Parameter(torch.randn(1, (frame_in_size // patch_size)**2, latent_dim) * .02)

        self.frame_in_size = frame_in_size
        self.frame_out_size = frame_out_size
        self.patch_size = patch_size
        self.latent_dim = latent_dim

    def forward(self, x: Tensor):
        """
        Input dims: [batch, time, height, width, channel]
        Output dims: [batch, time, height, width, channel]
        """
        b, t, h, w, c = x.shape

        x_patches = rearrange(x, "b t h w c -> (b t) c h w")
        x_patches = self.patchify(x_patches)
        x_patches = rearrange(x_patches, "(b t) e h w -> b (t h w) e", b=b, t=t)
        x_patches = self.pos_enc(x_patches)

        x_latent = rearrange(x, "b t h w c -> (b t) c h w")
        x_latent = self.frame_encoder(x_latent)
        x_latent = rearrange(x_latent, "(b t) e -> b t e", b=b, t=t)
        x_latent = self.pos_enc(x_latent)
        x_next_latent = self.latent_predictor(x_latent)

        #canvas = self.pos_enc(torch.ones(b, t, *self.frame_out_size, c))
        canvas = torch.ones(b, t, *self.frame_out_size, c)
        x_next_frame = self.latent_frame_decoder(x_next_latent, x_patches, canvas)

        return x_next_frame


class FrameEncoder(nn.Module):
    def __init__(
        self,
        frame_in_size: Union[int, Tuple[int, int]] = 256,
        patch_size: int = 16,
        depth: int = 8,
        latent_dim: int = 96,
        num_heads: int = 8,
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
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            num_classes=0,
            class_token=True,
        )

    def forward(self, x: Tensor):
        """
        Input dims: [batch * time, height, width, channel]
        Output dims: [batch * time, latent]
        """
        return self.backbone(x)


class LatentPredictor(nn.Module):
    def __init__(
        self,
        latent_dim: int = 768,
        num_heads: int = 8,
        window_size: int = 5,
    ) -> None:
        super().__init__()
        att = LocalAttention(causal=True, window_size=window_size)
        self.att_block = MultiHeadDispatch(latent_dim, num_heads, att)

    def forward(self, x):
        """
        Input dims: [batch, time, latent]
        Output dims: [batch, time, latent]
        """
        return self.att_block(x)


class LatentFrameDecoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_latent, x_patches, canvas):
        """
        Input dims:
            latent predictor: [batch, latent]
            frame patchify: [batch, time, height_patches, width_patches, latent]

        Output dims: [batch, time, height_out, width_out, channel]
        """
        print(f"x_latent: {x_latent.shape}\nx_patches: {x_patches.shape}\ncanvas: {canvas.shape}")
