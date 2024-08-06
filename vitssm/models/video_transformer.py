from typing import Union, Tuple


from einops import rearrange
from torch import nn
from torch import Tensor
from timm.models.vision_transformer import VisionTransformer
from xformers.components.positional_embedding import SinePositionalEmbedding


class LatentNextFramePrediction(nn.Module):
    def __init__(
        self,
        frame_in_size: Union[int, Tuple[int, int]] = 256,
        frame_out_size: Union[int, Tuple[int, int]] = 256,
        patch_size: int = 16,
        latent_dim: int = 768,
    ):
        super().__init__()
        self.frame_encoder = FrameEncoder()
        self.latent_predictor = LatentPredictor()
        self.latent_frame_decoder = LatentFrameDecoder()

        self.patchify = nn.Conv2d(3, latent_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_enc = SinePositionalEmbedding()

        self.frame_in_size = frame_in_size
        self.frame_out_size = frame_out_size
        self.patch_size = patch_size
        self.latent_dim = latent_dim

    def forward(x: Tensor):
        """
        Input dims: [batch, time, height, width, channel]
        """
        b, t, h, w, c = x.shape

        x_patches = rearrange(x, "b t h w c -> (b t) c h w")
        x_patches = self.patchify(x_patches)
        x_patches = rearrange(x_patches, "(b t) e h w -> b (t h w) e", b=b, t=t)
        x_patches = self.pos_enc(x_patches)

        x_latent = rearrange(x, "b t h w c -> (b t) h w c")
        x_latent = self.latent_encoder(x_latent)
        x_latent = rearrange(x, "(b t) e -> b t e", b=b, t=t)
        x_next_latent = self.latent_predictor(x_latent)


class FrameEncoder(nn.Module):
    def __init__(
        self,
        frame_in_size: Union[int, Tuple[int, int]] = 256,
        patch_size: int = 16,
        depth: int = 8,
        embed_dim: int = 96,
        num_heads: int = 8,
        mlp_ratio: float = 3.0,
        qkv_bias: bool = False,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()
        self.backbone = VisionTransformer(
            img_size=frame_in_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
        )

    def forward(self, x: Tensor):
        """
        Input dims: [batch * time, height, width, channel]
        Output dims: [batch * time, embedding]
        """
        return self.backbone(x)


class LatentPredictor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Input dims: [batch, time, embedding]
        Output dims: [batch, embedding]
        """
        pass


class LatentFrameDecoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_latent, x_patches):
        """
        Input dims:
            latent predictor: [batch, embedding]
            frame patchify: [batch, time, height_patches, width_patches, embedding]

        Output dims: [batch, time, height_out, width_out, channel]
        """
        pass
