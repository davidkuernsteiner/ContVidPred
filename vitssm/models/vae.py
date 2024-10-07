from typing import Any, Optional, Tuple
from torch import nn, Tensor
from diffusers.models import ModelMixin
from einops import rearrange
from pydantic import BaseModel

from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.autoencoders.vae import DecoderOutput


class VideoVAEConfig(BaseModel):
    in_channels: int = 3
    out_channels: int = 3
    down_block_types: tuple[str, ...] = ("DownEncoderBlock2D",)
    up_block_types: tuple[str, ...] = ("UpDecoderBlock2D",)
    block_out_channels: tuple[int, ...] = (64,)
    layers_per_block: int = 1
    act_fn: str = "silu"
    latent_channels: int = 4
    norm_num_groups: int = 32
    sample_size: int = 32
    scaling_factor: float = 0.18215
    shift_factor: Optional[float] = None
    latents_mean: Optional[tuple[float]] = None
    latents_std: Optional[tuple[float]] = None
    force_upcast: float = True
    use_quant_conv: bool = True
    use_post_quant_conv: bool = True
    mid_block_add_attention: bool = True
    
    
def VAETiny(**kwargs):
    return AutoencoderKL(
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels=(16, 16),
        layers_per_block=2,
        act_fn="silu",
        latent_channels=4,
        norm_num_groups=4,
        **kwargs,
    )


vae_models = {
    "vae-tiny": VAETiny
}


"""class VideoVAEWrapper(AutoencoderKL):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: tuple[str] = ("DownEncoderBlock2D",),
        up_block_types: tuple[str] = ("UpDecoderBlock2D",),
        block_out_channels: tuple[int] = (64,),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        sample_size: int = 32,
        scaling_factor: float = 0.18215,
        shift_factor: Optional[float] = None,
        latents_mean: Optional[tuple[float]] = None,
        latents_std: Optional[tuple[float]] = None,
        force_upcast: float = True,
        use_quant_conv: bool = True,
        use_post_quant_conv: bool = True,
        mid_block_add_attention: bool = True,
    ) -> None:
        super().__init__()
    
    def encode_video(self, x: Tensor) -> AutoencoderKLOutput:
        b, t, c, h, w = x.shape
        
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.encode(x, return_dict=True).latent_dist.sample().mul_(0.18215)
        x = rearrange(x, "(b t) c h w -> b t c h w", b=b, t=t)
        
        return x

    
    def decode_video(self, z: Tensor) -> DecoderOutput:
        b, t, e, h, w = z.shape
        
        z = rearrange(z, "b t e h w -> (b t) e h w")
        z = self.vae.encode(z)
        z = rearrange(z, "(b t) c h w -> b t c h w", b=b, t=t)
        
        return z
    
    def forward_video(self, x: Tensor, sample=False) -> DecoderOutput:
        pass
    
    def loss(self, x: DecoderOutput, x_hat: Tensor, mu: Tensor, logvar: Tensor) -> Tensor:
        pass
    
    def sample(self, mu: Tensor, logvar: Tensor) -> Tensor:
        pass"""
    