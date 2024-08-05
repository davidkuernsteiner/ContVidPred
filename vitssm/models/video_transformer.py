from typing import Union, Tuple

from torch import nn
from timm.models.vision_transformer import VisionTransformer


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
        self.pos_enc = None


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

    def forward(self, x):
        return self.backbone(x)


class LatentPredictor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


class LatentFrameDecoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass
