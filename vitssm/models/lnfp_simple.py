import torch
from einops import rearrange, repeat
from pydantic import BaseModel
from timm.models.vision_transformer import VisionTransformer
from torch import Tensor, nn
from xformers.components.attention import ScaledDotProduct
from xformers.components.multi_head_dispatch import MultiHeadDispatch

from .modules import LearnablePositionalEncoding, MixedCrossAttentionBlock


class LNFPSimpleConfig(BaseModel):
    frame_in_size: tuple[int, int] = (16, 16)
    frame_out_size: tuple[int, int] = (16, 16)
    n_frames: int = 10
    patch_size: int = 1
    latent_dim: int = 32
    n_blocks_encoder: int = 2
    n_heads_encoder: int = 1
    n_blocks_latent_predictor: int = 2
    n_heads_latent_predictor: int = 1
    n_blocks_decoder: int = 2
    n_heads_decoder: int = 1
    mlp_multiplier: int = 2
    pos_enc_dropout: float = 0.0
    residual_dropout: float = 0.0
    mlp_dropout: float = 0.0


class LNFPSimple(nn.Module):
    def __init__(
        self,
        frame_in_size: tuple[int, int] = (16, 16),
        frame_out_size: tuple[int, int] = (16, 16),
        n_frames: int = 10,
        patch_size: int = 1,
        latent_dim: int = 32,
        n_blocks_encoder: int = 2,
        n_heads_encoder: int = 1,
        n_blocks_latent_predictor: int = 2,
        n_heads_latent_predictor: int = 1,
        n_blocks_decoder: int = 2,
        n_heads_decoder: int = 1,
        mlp_multiplier: int = 2,
        pos_enc_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        mlp_dropout: float = 0.0,

    ):
        super().__init__()
        self.frame_encoder = FrameEncoder(
            frame_in_size=frame_in_size,
            patch_size=patch_size,
            latent_dim=latent_dim,
            n_blocks=n_blocks_encoder,
            n_heads=n_heads_encoder,
            mlp_ratio=mlp_multiplier,
        )
        self.latent_predictor = LatentPredictor(
            latent_dim=latent_dim,
            n_heads=n_heads_latent_predictor,
            n_blocks=n_blocks_latent_predictor,
            n_frames=n_frames,
            n_tokens=(frame_in_size[0] // patch_size) * (frame_in_size[1] // patch_size),
        )
        self.latent_frame_decoder = LatentFrameDecoder(
            n_blocks=n_blocks_decoder,
            latent_dim=latent_dim,
            n_heads=n_heads_decoder,
            residual_dropout=residual_dropout,
            mlp_dropout=mlp_dropout,
            mlp_multiplier=mlp_multiplier,
        )

        self.patchify = nn.Conv2d(
            3,
            latent_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.pos_enc_patches = LearnablePositionalEncoding(
            n_tokens=(frame_in_size[0] // patch_size) * (frame_in_size[1] // patch_size),
            latent_dim=latent_dim,
            p_dropout=pos_enc_dropout,
        )
        self.patch_token_decoder = nn.Linear(latent_dim, patch_size * patch_size * 3)

        self.frame_in_size = frame_in_size
        self.frame_out_size = frame_out_size
        self.n_frames = n_frames
        self.patch_size = patch_size
        self.latent_dim = latent_dim

    def forward(self, x: Tensor):
        """Input dims: [batch, time, height, width, channel]
        Output dims: [batch, time, height, width, channel]
        """
        b, t, h, w, c = x.shape
        hp_out, wp_out = self.frame_out_size[0] // self.patch_size, self.frame_out_size[1] // self.patch_size
        ph, pw = self.patch_size, self.patch_size

        x_patches = rearrange(x, "b t h w c -> (b t) c h w")
        x_patches = self.patchify(x_patches)
        x_patches = rearrange(x_patches, "(b t) e hp wp -> (b t) (hp wp) e", b=b, t=t)
        x_patches = self.pos_enc_patches(x_patches)
        # x_patches = rearrange(x_patches, "(b t) (hp wp) e -> b t (hp wp) e", b=b, t=t, wp=wp, hp=hp)

        x_latent = rearrange(x, "b t h w c -> (b t) c h w")
        x_latent = self.frame_encoder(x_latent)
        x_latent = rearrange(x_latent, "(b t) nt e -> b (t nt) e", b=b, t=t, nt=hp_out * wp_out)
        x_next_latent = self.latent_predictor(x_latent)
        x_next_latent = rearrange(x_next_latent, "b (t nt) e -> (b t) nt e", t=t, nt=hp_out * wp_out)

        canvas = self.pos_enc_patches(torch.ones(b, hp_out * wp_out, self.latent_dim, dtype=x.dtype, device=x.device))
        canvas = repeat(canvas, "b (hp wp) e -> (b t) (hp wp) e", hp=hp_out, wp=wp_out, t=t)
        x_next_frame = self.latent_frame_decoder(canvas, x_patches, x_next_latent)
        x_next_frame = self.patch_token_decoder(x_next_frame)
        x_next_frame = rearrange(
            x_next_frame,
            "(b t) (hp wp) (ph pw c) -> b t (hp ph) (wp pw) c",
            t=t,
            hp=hp_out,
            wp=hp_out,
            ph=ph,
            pw=pw,
            c=3,
        )

        return x_next_frame


class FrameEncoder(nn.Module):
    def __init__(
        self,
        frame_in_size: tuple[int, int] = (16, 16),
        patch_size: int = 1,
        latent_dim: int = 32,
        n_blocks: int = 2,
        n_heads: int = 1,
        mlp_ratio: int = 2,
        qkv_bias: bool = False,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.backbone = VisionTransformer(
            img_size=frame_in_size[0],
            patch_size=patch_size,
            embed_dim=latent_dim,
            depth=n_blocks,
            num_heads=n_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            num_classes=0,
            class_token=False,
            global_pool="",
        )

    def forward(self, x: Tensor) -> Tensor:
        """Input dims: [batch * time, channel, height, width]
        Output dims: [batch * time, height // patchsize * width // patchsize, latent]
        """
        return self.backbone(x)


class LatentPredictor(nn.Module):
    def __init__(
        self,
        latent_dim: int = 64,
        n_heads: int = 1,
        n_blocks: int = 4,
        n_frames: int = 30,
        n_tokens: int = 265,
        window_size: int = 10,
    ) -> None:
        super().__init__()
        attention_mechanism = ScaledDotProduct(causal=True)

        self.pos_enc = LearnablePositionalEncoding(n_tokens=n_frames * n_tokens, latent_dim=latent_dim)
        self.blocks = nn.Sequential(
            *(MultiHeadDispatch(latent_dim, n_heads, attention_mechanism) for _ in range(n_blocks)),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Input dims: [batch, time * height // patchsize * width // patchsize, latent]
        Output dims: [batch, time * height // patchsize * width // patchsize, latent]
        """
        return self.blocks(self.pos_enc(x))


class LatentFrameDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 64,
        n_blocks: int = 4,
        n_heads: int = 1,
        mlp_multiplier: int = 2,
        residual_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            *(
                MixedCrossAttentionBlock(
                    latent_dim=latent_dim,
                    n_heads=n_heads,
                    residual_dropout=residual_dropout,
                    mlp_dropout=mlp_dropout,
                    mlp_multiplier=mlp_multiplier,
                )
                for _ in range(n_blocks)
            ),
        )

    def forward(self, x_query: Tensor, x_patches: Tensor, x_latent: Tensor) -> Tensor:
        """Input dims:
            x_query: [batch * time, height * width, latent]
            x_patches: [batch, time, height_patches, width_patches, latent]
            x_latent: [batch, time * height // patchsize * width // patchsize, latent]

        Output dims: [batch, time, height_out, width_out, channel]
        """
        inputs = x_query, x_patches, x_latent
        x_query, _, _ = self.blocks(inputs)

        return x_query
