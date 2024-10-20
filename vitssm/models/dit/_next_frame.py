from typing import Any, Literal
from torch import nn, Tensor
import torch
from pydantic import BaseModel
from einops import rearrange

from ._dit import DiT, DiT_models
from ..diffusion import create_diffusion
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

from ..vae import VideoVAEConfig, vae_models

    
    
class DiffusionConfig(BaseModel):
    timestep_respacing: str = ""
    noise_schedule: str = "linear"
    use_kl: bool = False
    sigma_small: bool = False
    predict_xstart: bool = False
    learn_sigma: bool = True
    rescale_learned_sigmas: bool = False
    diffusion_steps: int = 1000
    

class NextFrameDiTModelConfig(BaseModel):
    vae_type: Literal["vae-tiny", "vae-small", "vae-base", "vae-large"] = "vae-tiny"
    vae_kwargs: dict[str, Any] = {}
    vae_checkpoint_path: str | None = None
    dit_type: Literal['DiT_T_2', 'DiT_T_4', 'DiT_M_1', 'DiT_M_2'] = "DiT_T_2"
    dit_kwargs: dict[str, Any] = {}
    timestep_respacing: str = ""
    diffusion_steps: int = 1000


class NextFrameDiTModel(nn.Module):
    def __init__(
        self,
        vae_type: Literal["vae-tiny", "vae-small", "vae-base", "vae-large"] = "vae-tiny",
        vae_kwargs: dict = {},
        vae_checkpoint_path: str | None = None,
        dit_type: Literal['DiT_T_2', 'DiT_T_4', 'DiT_M_1', 'DiT_M_2'] = "DiT_T_2",
        dit_kwargs: dict = {},
        timestep_respacing: str = "",
        diffusion_steps: int = 1000,
    ):
        super().__init__()
        
        self.vae = vae_models[vae_type](**vae_kwargs)
        if vae_checkpoint_path is not None:
            self.vae.load_state_dict(torch.load(vae_checkpoint_path)["ema" if "ema" in vae_checkpoint_path else "model"])
        self.vae.requires_grad_(False)
            
        self.dit = DiT_models[dit_type](**dit_kwargs)
        self.dit.initialize_weights()
        
        self.diffusion = create_diffusion(
            **DiffusionConfig(
                timestep_respacing=timestep_respacing,
                diffusion_steps=diffusion_steps,
            ).model_dump()
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def forward_train(self, _context_frames: Tensor, _next_frame: Tensor) -> float:
        x = torch.cat((_context_frames, _next_frame), dim=1)
        b, t, _, _, _ = x.shape
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        
        with torch.no_grad():
            x = self.vae.encode(x).latent_dist.sample().mul_(0.18215)
            
        x = rearrange(x, '(b t) c h w -> b t c h w', b=b)
        x = rearrange(x, "b t c h w -> b (t c) h w")
        x_context, x = torch.split(
            x,
            [int((t - 1) * (x.shape[1] / t)),  int(x.shape[1] / t)],
            dim=1
        )
        
        model_kwargs = dict(x_context=x_context, y=None)
        t = torch.randint(0, self.diffusion.num_timesteps, (x.shape[0],), device=self.device)
        
        loss_dict = self.diffusion.training_losses(self.dit, x, t, model_kwargs=model_kwargs)
        loss = loss_dict["loss"].mean()
        
        return loss
    
    def sample() -> Tensor:
        pass