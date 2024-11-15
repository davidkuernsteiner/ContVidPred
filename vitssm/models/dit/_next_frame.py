from typing import Any, Literal
from torch import nn, Tensor
import torch
from pydantic import BaseModel
from einops import rearrange

from ._dit import DiT, DiT_models
from ..diffusion import create_diffusion, SpacedDiffusion, DiffusionConfig
from ...utils import ema_to_model_state_dict
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

from ..vae import VideoVAEConfig, vae_models
    

class NextFrameDiTModelConfig(BaseModel):
    vae_type: Literal["vae-tiny", "vae-small", "vae-base", "vae-large"] = "vae-tiny"
    vae_kwargs: dict[str, Any] = {}
    vae_checkpoint_path: str | None = None
    latent_scale_factor: float = 0.18215
    dit_type: Literal["DiT_M_1", "DiT_M_2", "DiT_T_1", "DiT_T_2"] = "DiT_T_2"
    dit_kwargs: dict[str, Any] = {}
    diffusion_steps: int = 1000
    timestep_respacing: str = "trailing25"
    predict_target: Literal["epsilon", "velocity", "xstart", "xprev"] = "velocity"
    rescale_betas_zero_snr: bool = True
    device: Literal["cpu", "cuda"] = "cuda"
    
    
class NextFrameDiTModel(nn.Module):
    def __init__(
        self,
        vae_type: Literal["vae-tiny", "vae-small", "vae-base", "vae-large"] = "vae-tiny",
        vae_kwargs: dict = {},
        vae_checkpoint_path: str | None = None,
        latent_scale_factor: float = 0.18215,
        dit_type: Literal["DiT_M_1", "DiT_M_2", "DiT_T_1", "DiT_T_2"] = "DiT_T_2",
        dit_kwargs: dict = {},
        diffusion_steps: int = 1000,
        timestep_respacing: str = "trailing4",
        predict_target: Literal["epsilon", "velocity", "xstart", "xprev"] = "velocity",
        rescale_betas_zero_snr: bool = True,
        device: Literal["cpu", "cuda"] = "cuda",
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
            
        self.dit = DiT_models[dit_type](**dit_kwargs)
        
        self.train_diffusion = create_diffusion(
            **DiffusionConfig(
                timestep_respacing="",
                predict_target=predict_target,
                rescale_betas_zero_snr=rescale_betas_zero_snr,
                diffusion_steps=diffusion_steps
            ).model_dump()
        )
        self.sampling_diffusion = create_diffusion(
            **DiffusionConfig(
                timestep_respacing=timestep_respacing,
                predict_target=predict_target,
                rescale_betas_zero_snr=rescale_betas_zero_snr,
                diffusion_steps=diffusion_steps  
            ).model_dump()
        )
        
        self.device = torch.device(device)
        self.scale_factor = latent_scale_factor
    
    def forward_train(self, _context_frames: Tensor, _next_frame: Tensor) -> float:
        x = torch.cat((_context_frames, _next_frame), dim=1)
        b, t, _, _, _ = x.shape
        
        # Encode frames
        if self.vae is not None:
            x = rearrange(x, 'b t c h w -> (b t) c h w')
            with torch.no_grad():
                x = self.vae.encode(x).latent_dist.sample().mul_(self.scale_factor)    
            x = rearrange(x, '(b t) c h w -> b t c h w', b=b)
        
        x = rearrange(x, "b t c h w -> b (t c) h w")
        x_context, x = torch.split(
            x,
            [int((t - 1) * (x.shape[1] / t)),  int(x.shape[1] / t)],
            dim=1
        )
        
        model_kwargs = dict(context=x_context, y=None)
        t = torch.randint(0, self.train_diffusion.num_timesteps, (x.shape[0],), device=self.device)
        
        loss_dict = self.train_diffusion.training_losses(self.dit, x, t, model_kwargs=model_kwargs)
        loss = loss_dict["loss"].mean()
        
        return loss
    
    def rollout_frames(self, x_context: Tensor, n_steps: int) -> Tensor:
        """
        Rollout frames using the model.
        x_context: [N, T, C, H, W]
        """
        n, t, c, h, w = x_context.shape
        if self.vae is not None:
            x_context = rearrange(x_context, "n t c h w -> (n t) c h w")
            x_context = self.vae.encode(x_context).latent_dist.sample().mul_(self.scale_factor)
            x_context = rearrange(x_context, "(n t) c h w -> n t c h w", n=n)
        
        frames = []
        for _ in range(n_steps):
            frame = self._sample_frame(x_context).unsqueeze(1)
            frames.append(frame.clone())
            x_context = torch.cat((x_context[:, 1:], frame), dim=1)
        
        frames = torch.cat(frames, dim=1)
        
        if self.vae is not None:
            frames = rearrange(frames, "n t c h w -> (n t) c h w")
            frames = self.vae.decode(frames / self.scale_factor).sample
            frames = rearrange(frames, "(n t) c h w -> n t c h w", n=n)
        
        return frames
    
    def _sample_frame(self, x_context: Tensor) -> Tensor:
        n, t, c, h, w = x_context.shape
        x = torch.randn(n, c, h, w, device=self.device)
        x_context = rearrange(x_context, "n t c h w -> n (t c) h w")
        
        model_kwargs = dict(context=x_context, y=None)
        
        samples = self.sampling_diffusion.ddim_sample_loop(
            self.dit, x.shape, x, clip_denoised=True, model_kwargs=model_kwargs, progress=False, device=self.device
        )
        
        return samples