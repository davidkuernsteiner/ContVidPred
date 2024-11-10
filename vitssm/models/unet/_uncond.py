from typing import Any, Literal
from torch import nn, Tensor
import torch
from pydantic import BaseModel
from einops import rearrange

from ._unet import UNet_models
from ..diffusion import create_diffusion, SpacedDiffusion, DiffusionConfig
from ...utils import ema_to_model_state_dict
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

from ..vae import VideoVAEConfig, vae_models


class UncondUNetModelConfig(BaseModel):
    vae_type: Literal["vae-tiny", "vae-small", "vae-base", "vae-large"] = "vae-tiny"
    vae_kwargs: dict[str, Any] = {}
    vae_checkpoint_path: str | None = None
    latent_scale_factor: float = 0.18215
    unet_type: Literal["UNet_B", "UNet_S", "UNet_T", "UNet_M"] = "UNet_M"
    unet_kwargs: dict[str, Any] = {}
    timestep_respacing: str = ""
    diffusion_steps: int = 1000
    device: Literal["cpu", "cuda"] = "cuda"


class UncondUNetModel(nn.Module):
    def __init__(
        self,
        vae_type: Literal["vae-tiny", "vae-small", "vae-base", "vae-large"] = "vae-tiny",
        vae_kwargs: dict = {},
        vae_checkpoint_path: str | None = None,
        latent_scale_factor: float = 0.18215,
        unet_type: Literal["UNet_B", "UNet_S", "UNet_T", "UNet_M"] = "UNet_M",
        unet_kwargs: dict = {},
        timestep_respacing: str = "",
        device: Literal["cpu", "cuda"] = "cuda"
    ):
        super().__init__()
        
        if vae_checkpoint_path is not None:
            self.vae = vae_models[vae_type](**vae_kwargs)
            checkpoint = torch.load(vae_checkpoint_path)
            self.vae.load_state_dict(
                ema_to_model_state_dict(checkpoint["ema"]) if "ema" in checkpoint.keys() else checkpoint["model"]
            )
            self.vae.requires_grad_(False)
        else:
            self.vae = None
            
        self.unet = UNet_models[unet_type](**unet_kwargs)
        
        self.train_diffusion = create_diffusion(
            **DiffusionConfig().model_dump()
        )
        self.sampling_diffusion = create_diffusion(
            **DiffusionConfig(
                timestep_respacing=timestep_respacing,    
            ).model_dump()
        )
        
        self.device = torch.device(device)
        self.scale_factor = latent_scale_factor
    
    def forward_train(self, x: Tensor) -> float:
        b, t, _, _, _ = x.shape
        
        if self.vae is not None:
            with torch.no_grad():
                x = self.vae.encode(x).latent_dist.sample().mul_(self.scale_factor)    
        
        model_kwargs = None
        t = torch.randint(0, self.diffusion.num_timesteps, (x.shape[0],), device=self.device)
        
        loss_dict = self.diffusion.training_losses(self.unet, x, t, model_kwargs=model_kwargs)
        loss = loss_dict["loss"].mean()
        
        return loss
    
    def sample(self, n_samples: int) -> Tensor:
        h, w = self.unet.sample_size
        x = torch.randn(n_samples, self.unet.out_channels, h, w, device=self.device)
        x_context = rearrange(x_context, "n t c h w -> n (t c) h w")
        
        samples = self.diffusion.ddim_sample_loop(
            self.unet, x.shape, x, clip_denoised=True, model_kwargs=None, progress=False, device=self.device
        )
        
        return samples