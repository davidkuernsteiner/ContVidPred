from typing import Any, Literal
from torch import nn, Tensor
import torch
from pydantic import BaseModel

from latte.models.latte import Latte
from latte.diffusion import create_diffusion
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

from .vae import VideoVAEConfig, vae_models


class LatteConfig(BaseModel):
    input_size: int = 32
    patch_size: int = 2
    in_channels: int = 4
    hidden_size: int = 1152
    depth: int = 28
    num_heads: int = 16
    mlp_ratio: float = 4.0
    num_frames: int = 16
    class_dropout_prob: float = 0.1
    num_classes: int = 1000
    learn_sigma: bool = True
    extras: int = 1
    attention_mode: str = 'math'
    

class LatteDiffusionModelConfig(BaseModel):
    vae_type: Literal["vae-tiny", "vae-small", "vae-base", "vae-large"] = "vae-tiny"
    vae_kwargs: dict[str, Any] = {}
    latte_type: Literal["latte-tiny", "latte-small", "latte-base", "latte-large"] = "latte-tiny"
    latte_kwargs: dict[str, Any] = {}


class LatteDiffusionModel(nn.Module):
    def __init__(
        self,
        vae_type: Literal["vae-tiny", "vae-small", "vae-base", "vae-large"] = "vae-tiny",
        vae_kwargs: dict = {},
        latte_type: Literal["latte-tiny", "latte-small", "latte-base", "latte-large"] = "latte-tiny",
        latte_kwargs: dict = {},
        diffusion_steps: int = 100
    ):
        super().__init__()
        
        self.vae = vae_models[vae_type](**vae_kwargs)
        self.latte = latte_models[latte_type](**latte_kwargs)
        self.diffusion = create_diffusion(
            timestep_respacing="",
            diffusion_steps=diffusion_steps,
        )
        
        
        

def LatteTiny(**kwargs):
    return Latte(depth=2, hidden_size=32, patch_size=2, num_heads=1, **kwargs)


latte_models = {
    "latte-tiny": LatteTiny
}