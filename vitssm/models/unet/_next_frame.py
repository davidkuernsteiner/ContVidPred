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


class NextFrameUNetModelConfig(BaseModel):
    vae_type: Literal["vae-tiny", "vae-small", "vae-base", "vae-large"] = "vae-tiny"
    vae_kwargs: dict[str, Any] = {}
    vae_checkpoint_path: str | None = None
    latent_scale_factor: float = 0.18215
    unet_type: Literal["UNet_B", "UNet_S", "UNet_T", "UNet_M"] = "UNet_M"
    unet_kwargs: dict[str, Any] = {}
    diffusion_steps: int = 1000
    timestep_respacing: str = ""
    noise_augmentation_scale: float = 0.7
    noise_augmentation_buckets: int = 10
    cfg_dropout: float = 0.1
    cfg_scale: float = 7.5
    cfg_rescale_factor: float = 0.7
    device: Literal["cpu", "cuda"] = "cuda"


class NextFrameUNetModel(nn.Module):
    def __init__(
        self,
        vae_type: Literal["vae-tiny", "vae-small", "vae-base", "vae-large"] = "vae-tiny",
        vae_kwargs: dict = {},
        vae_checkpoint_path: str | None = None,
        latent_scale_factor: float = 0.18215,
        unet_type: Literal["UNet_B", "UNet_S", "UNet_T", "UNet_M"] = "UNet_M",
        unet_kwargs: dict = {},
        diffusion_steps: int = 1000,
        timestep_respacing: str = "trailing4",
        noise_augmentation_scale: float = 0.7,
        noise_augmentation_buckets: int = 10,
        cfg_dropout: float = 0.1,
        cfg_scale: float = 7.5,
        cfg_rescale_factor: float = 0.7,
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
            
        self.unet = UNet_models[unet_type](**unet_kwargs)
        
        self.train_diffusion = create_diffusion(
            **DiffusionConfig(
                timestep_respacing="",
                predict_target="velocity",
                rescale_betas_zero_snr=True,
                diffusion_steps=diffusion_steps
            ).model_dump()
        )
        self.sampling_diffusion = create_diffusion(
            **DiffusionConfig(
                timestep_respacing=timestep_respacing,
                predict_target="velocity",
                rescale_betas_zero_snr=True,
                diffusion_steps=diffusion_steps  
            ).model_dump()
        )
        
        self.device = torch.device(device)
        self.scale_factor = latent_scale_factor
        self.noise_augmentation_scale = noise_augmentation_scale
        self.noise_augmentation_buckets = noise_augmentation_buckets
        self.cfg_dropout = cfg_dropout
        self.cfg_scale = cfg_scale
        self.cfg_rescale_factor = cfg_rescale_factor
    
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
        
        # Add noise augmentation
        alpha_cond_aug = torch.rand(b, device=self.device)[:, None, None, None] * self.noise_augmentation_scale  
        x_context, alpha_buckets = self._cond_augmentation_from_alpha(x_context, alpha_cond_aug)
        
        # Dropout for cfg
        drop_idcs = torch.rand(b, device=self.device) < self.cfg_dropout
        x_context[drop_idcs] = torch.zeros_like(x_context[drop_idcs])
        alpha_buckets[drop_idcs] = self.noise_augmentation_buckets
        
        model_kwargs = dict(context=x_context, class_labels=alpha_buckets)
        t = torch.randint(0, self.diffusion.num_timesteps, (x.shape[0],), device=self.device)
        
        loss_dict = self.diffusion.training_losses(self.unet, x, t, model_kwargs=model_kwargs)
        loss = loss_dict["loss"].mean()
        
        return loss
    
    def rollout_frames(self, x_context: Tensor, n_steps: int, alpha_cond_aug: float) -> Tensor:
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
            frame = self._sample_frame(x_context, alpha_cond_aug).unsqueeze(1)
            frames.append(frame)
            x_context = torch.cat((x_context[:, 1:], frame), dim=1)
        
        frames = torch.cat(frames, dim=1)
        
        if self.vae is not None:
            frames = rearrange(frames, "n t c h w -> (n t) c h w")
            frames = self.vae.decode(frames / self.scale_factor).sample
            frames = rearrange(frames, "(n t) c h w -> n t c h w", n=n)
        
        return frames
    
    def _cond_augmentation_from_alpha(self, x_cond: Tensor, alpha_cond_aug: Tensor) -> tuple[Tensor, Tensor]:
        cond_noise = torch.randn_like(x_cond, device=self.device)
        x_cond = alpha_cond_aug.sqrt() * x_cond + (1 - alpha_cond_aug).sqrt() * cond_noise
        bins = torch.linspace(0., self.noise_augmentation_scale, self.noise_augmentation_buckets + 1, device=self.device)
        alpha_buckets = torch.bucketize(alpha_cond_aug, bins, right=True) - 1
        
        return x_cond, alpha_buckets
    
    def _sample_frame(self, x_context: Tensor, alpha_cond_aug: float) -> Tensor:
        n, t, c, h, w = x_context.shape
        x = torch.randn(n, c, h, w, device=self.device)
        x_context = rearrange(x_context, "n t c h w -> n (t c) h w")
        
        #Noise Augmentation
        x_context, alpha_buckets = self._cond_augmentation_from_alpha(x_context, torch.ones(n, 1, 1, 1, device=self.device) * alpha_cond_aug)
        
        # cfg
        x = torch.cat([x, x], 0)
        alpha_buckets_null = torch.tensor([self.noise_augmentation_buckets] * n, device=self.device)
        alpha_buckets = torch.cat([alpha_buckets, alpha_buckets_null], 0)
        
        model_kwargs = dict(context=x_context, class_labels=alpha_buckets, cfg_scale=self.cfg_scale, cfg_rescale_factor=self.cfg_rescale_factor)
        
        samples = self.diffusion.ddim_sample_loop(
            self.unet.forward_with_cfg, x.shape, x, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=self.device
        )
        
        return samples
