from typing import Any, Literal
from torch import nn, Tensor
import torch
from pydantic import BaseModel
from einops import rearrange

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


def LatteTiny(**kwargs):
    return Latte(depth=2, hidden_size=32, patch_size=2, num_heads=1, **kwargs)


latte_models = {
    "latte-tiny": LatteTiny
}
    
    
class DiffusionConfig(BaseModel):
    timestep_respacing: str = ""
    noise_schedule: str = "linear"
    use_kl: bool = False
    sigma_small: bool = False
    predict_xstart: bool = False
    learn_sigma: bool = True
    rescale_learned_sigmas: bool = False
    diffusion_steps: int = 1000
    

class LatteDiffusionModelConfig(BaseModel):
    vae_type: Literal["vae-tiny", "vae-small", "vae-base", "vae-large"] = "vae-tiny"
    vae_kwargs: dict[str, Any] = {}
    vae_checkpoint_path: str | None = None
    latte_type: Literal["latte-tiny", "latte-small", "latte-base", "latte-large"] = "latte-tiny"
    latte_kwargs: dict[str, Any] = {}
    timestep_respacing: str = ""
    diffusion_steps: int = 100
    


class LatteDiffusionModel(nn.Module):
    def __init__(
        self,
        vae_type: Literal["vae-tiny", "vae-small", "vae-base", "vae-large"] = "vae-tiny",
        vae_kwargs: dict = {},
        vae_checkpoint_path: str | None = None,
        latte_type: Literal["latte-tiny", "latte-small", "latte-base", "latte-large"] = "latte-tiny",
        latte_kwargs: dict = {},
        timestep_respacing: str = "",
        diffusion_steps: int = 100,
        gradient_accumulation_steps: int = 1,
    ):
        super().__init__()
        
        self.vae = vae_models[vae_type](**vae_kwargs)
        if vae_checkpoint_path is not None:
            self.vae.load_state_dict(torch.load(vae_checkpoint_path)["ema" if "ema" in vae_checkpoint_path else "model"])
        self.vae.requires_grad_(False)
            
        self.latte = latte_models[latte_type](**latte_kwargs)
        
        self.diffusion = create_diffusion(
            **DiffusionConfig(
                timestep_respacing=timestep_respacing,
                diffusion_steps=diffusion_steps,
            )
        )
        
        self.gradient_accumulation_steps = gradient_accumulation_steps
    
    def forward_train(self, x: Tensor) -> float:
        with torch.no_grad():
            b, t, c, h, w = x.shape
            x = rearrange(x, 'b t c h w -> (b t) c h w')
            x = self.vae.encode(x).latent_dist.sample().mul_(0.18215)
            x = rearrange(x, '(b t) c h w -> b t c h w', b=b)
            
        t = torch.randint(0, self.diffusion.num_timesteps, (x.shape[0],), device=self.device)
        loss_dict = self.diffusion.training_losses(self.latte, x, t, dict(y=None))
        loss = loss_dict["loss"].mean() / self.gradient_accumulation_steps
        
        return loss
    

    def sample() -> Tensor:
        pass
    