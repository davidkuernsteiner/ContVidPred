from typing import Any, Literal
from torch import nn, Tensor
import torch
from pydantic import BaseModel
from einops import rearrange

from ._dit import DiT, DiT_models
from ..diffusion import create_diffusion, SpacedDiffusion
from ...utils import ema_to_model_state_dict
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
            checkpoint = torch.load(vae_checkpoint_path)
            self.vae.load_state_dict(
                ema_to_model_state_dict(checkpoint["ema"]) if "ema" in checkpoint.keys() else checkpoint["model"]
            )
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
            z = self.vae.encode(x).latent_dist.sample().mul_(0.18215)
            
        z = rearrange(z, '(b t) c h w -> b t c h w', b=b)
        z = rearrange(z, "b t c h w -> b (t c) h w")
        z_context, z = torch.split(
            z,
            [int((t - 1) * (z.shape[1] / t)),  int(z.shape[1] / t)],
            dim=1
        )
        
        model_kwargs = dict(x_context=z_context, y=None)
        t = torch.randint(0, self.diffusion.num_timesteps, (z.shape[0],), device=self.device)
        
        loss_dict = self.diffusion.training_losses(self.dit, z, t, model_kwargs=model_kwargs)
        loss = loss_dict["loss"].mean()
        
        return loss
    
    def sample_frame_latents(self, z_context: Tensor) -> Tensor:
        n, t, c_z, h_z, w_z = z_context.shape
        z = torch.randn(n, c_z, h_z, w_z, device=self.device)
        z_context = rearrange(z_context, "n t c h w -> n (t c) h w")
        model_kwargs = dict(x_context=z_context, y=None)
        
        samples = self.diffusion.ddim_sample_loop(
            self.dit, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=self.device
        )
        
        return samples
    
    def rollout_frames(self, x_context: Tensor, n_steps: int) -> Tensor:
        """
        Rollout frames using the model.
        x_context: [N, T, C, H, W]
        """
        n, t, c, h, w = x_context.shape
        x_context = rearrange(x_context, "n t c h w -> (n t) c h w")
        z_context = self.vae.encode(x_context).latent_dist.sample().mul_(0.18215)
        z_context = rearrange(z_context, "(n t) c h w -> n t c h w", n=n)
        
        frames = []
        for _ in range(n_steps):
            frame = self.sample_frame_latents(z_context).unsqueeze(1)
            frames.append(frame)
            z_context = torch.cat((z_context[:, 1:], frame), dim=1)
        
        frames = torch.cat(frames, dim=1)
        frames = rearrange(frames, "n t c h w -> (n t) c h w")
        frames = self.vae.decode(frames / 0.18215).sample
        frames = rearrange(frames, "(n t) c h w -> n t c h w", n=n)
        
        return frames