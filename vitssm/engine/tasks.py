from typing import Any, Union

import torch
import numpy as np
from omegaconf.dictconfig import DictConfig
from torch import Tensor, nn
import torchvision.transforms.v2.functional as F
from einops import rearrange, repeat

from wandb.sdk.wandb_run import Run
import wandb

from . import ModelEngine
from ..models import UncondUNetModel, NextFrameUNetModel, NextFrameDiTModel
from ..models.vae import frange_cycle_linear
from ..utils.metrics import RolloutMetricCollectionWrapper, AutoEncoderMetricCollectionWrapper, VideoAutoEncoderMetricCollectionWrapper, VariableResVideoAutoEncoderMetricCollectionWrapper
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
#from latte.utils import clip_grad_norm_	
    
    
class VAEEngine(ModelEngine):
    
    def __init__(self, model: AutoencoderKL, run_object: DictConfig, resume: bool = False) -> None:
        super().__init__(model, run_object, resume=resume)
        if self.config.model.get("use_beta_schedule", True):
            self.betas = frange_cycle_linear(self.steps, start=0.0, stop=self.config.model.get("beta", 1.0), n_cycle=4, ratio=0.5)
        else:
            self.betas = np.ones(self.steps) * self.config.model.get("beta", 1.0)
    
    def _train_step(self, _x: Tensor, _y: Tensor) -> dict[str, float]:
        beta = self.betas[self.state["step"] - 1]
        self.optimizer.zero_grad()
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
            _posterior = self.model.encode(_x).latent_dist
            _recon = self.model.decode(_posterior.sample()).sample
            
            _recon_loss = self.criterion(_recon, _x)
            _kl_loss = _posterior.kl().mean()
            _loss = _recon_loss + beta * _kl_loss
            
        self.scaler.scale(_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        if (self.scheduler is not None) and self.scheduler_step_on_batch:
            self.scheduler.step()
        
        return {"loss": _loss.item(), "recon_loss": _recon_loss.item(), "kl_loss": _kl_loss.item(), "beta": beta}
    
    @torch.no_grad()
    def _eval_step(self, _x: Tensor, _y: Tensor) -> dict[str, float]:
        beta = self.betas[self.state["step"] - 1]
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
            _posterior = self.eval_model.encode(_x).latent_dist
            _recon = self.eval_model.decode(_posterior.sample()).sample
            
            _recon_loss = self.criterion(_recon, _x)
            _kl_loss = _posterior.kl().mean()
            _loss = _recon_loss + beta * _kl_loss
        
        if self.metrics is not None:
            self.metrics.update(_recon, _x)
        
        return {"loss": _loss.item(), "recon_loss": _recon_loss.item(), "kl_loss": _kl_loss.item()}
    
    
class NextFrameDiTEngine(ModelEngine):
    def __init__(self, model: NextFrameDiTModel, run_object: DictConfig, resume: bool = False) -> None:
        super().__init__(model, run_object, resume=resume)
        self.metrics = RolloutMetricCollectionWrapper(self.metrics) if self.metrics is not None else None
    
    def _train_step(self, _context_frames: Tensor, _next_frame: Tensor) -> dict[str, float]:
        self.optimizer.zero_grad()
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
            _loss = self.model.forward_train(_context_frames, _next_frame)
            
        self.scaler.scale(_loss).backward()           
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        if (self.scheduler is not None) and self.scheduler_step_on_batch:
            self.scheduler.step()
        
        return {"loss": _loss.item()}
    
    @torch.no_grad()
    def _eval_step(self, _x: Tensor, _y: Tensor) -> dict[str, float]:
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
            _frames = self.eval_model.rollout_frames(_x, _y.shape[1])
        self.metrics.update(_frames, _y)
        
        return {}
    
    @staticmethod
    def _log_eval(epoch: int, step: int, eval_outs: dict) -> None:
        wandb.log(eval_outs, step=step)


class NextFrameUNetEngine(ModelEngine):
    def __init__(self, model: NextFrameUNetModel, run_object: DictConfig, resume: bool = False) -> None:
        super().__init__(model, run_object, resume=resume)
        self.metrics = RolloutMetricCollectionWrapper(self.metrics) if self.metrics is not None else None
    
    def _train_step(self, _context_frames: Tensor, _next_frame: Tensor) -> dict[str, float]:
        self.optimizer.zero_grad()
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
            _loss = self.model.forward_train(_context_frames, _next_frame)
            
        self.scaler.scale(_loss).backward()           
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        if (self.scheduler is not None) and self.scheduler_step_on_batch:
            self.scheduler.step()
        
        return {"loss": _loss.item()}
    
    @torch.no_grad()
    def _eval_step(self, _x: Tensor, _y: Tensor) -> dict[str, float]:
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
            _frames = self.eval_model.rollout_frames(
                _x,
                _y.shape[1],
                alpha_cond_aug=self.config.model.get("sampling_alpha_cond_aug", 0.0)
            )
        self.metrics.update(_frames, _y)
        
        return {}
    
    @staticmethod
    def _log_eval(epoch: int, step: int, eval_outs: dict) -> None:
        wandb.log(eval_outs, step=step)


class AutoEncoderUPTEngine(ModelEngine):   
    def __init__(self, model: nn.Module, run_object: DictConfig, resume: bool = False) -> None:
        super().__init__(model, run_object, resume=resume)
        self.metrics = AutoEncoderMetricCollectionWrapper(self.metrics) if self.metrics is not None else None
    
    def _train_step(self, _x: Tensor, _y: Tensor) -> dict[str, float]:
        self.optimizer.zero_grad()
        b, t, c, h, w = _x.shape
        
        _x = rearrange(_x, "b t c h w -> (b t) c h w")
        output_pos = rearrange(
            torch.stack(torch.meshgrid([torch.arange(h), torch.arange(h)], indexing="ij")),
            "ndim height width -> (height width) ndim",
        ).float().to(self.device)
        output_pos = output_pos / (h - 1)  * 1000
        
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
            _pred = self.model(_x, output_pos=repeat(output_pos, "... -> b ...", b=b*t))
            _loss = self.criterion(_pred, _x)
            
        self.scaler.scale(_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        if (self.scheduler is not None) and self.scheduler_step_on_batch:
            self.scheduler.step()
        
        return {"loss": _loss.item()}
    
    @torch.no_grad()
    def _eval_step(self, _x: Tensor, _y: Tensor) -> dict[str, float]:
        b, t, c, h, w = _x.shape
        
        _x = rearrange(_x, "b t c h w -> (b t) c h w")
        output_pos = rearrange(
            torch.stack(torch.meshgrid([torch.arange(h), torch.arange(h)], indexing="ij")),
            "ndim height width -> (height width) ndim",
        ).float().to(self.device)
        output_pos = output_pos / (h - 1) * 1000
        
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
            _pred = self.eval_model(_x, output_pos=repeat(output_pos, "... -> b ...", b=b*t))
        
        if self.metrics is not None:
            self.metrics.update(_pred, _x)
        
        return {}
    
    @staticmethod
    def _log_eval(epoch: int, step: int, eval_outs: dict) -> None:
        wandb.log(eval_outs, step=step)


class VideoAutoEncoderUPTEngine(ModelEngine):   
    def __init__(self, model: nn.Module, run_object: DictConfig, resume: bool = False) -> None:
        super().__init__(model, run_object, resume=resume)
        self.metrics = VideoAutoEncoderMetricCollectionWrapper(self.metrics) if self.metrics is not None else None
    
    def _train_step(self, _x: Tensor, _y: Tensor) -> dict[str, float]:
        self.optimizer.zero_grad()
        b, t, c, h, w = _x.shape
        
        output_pos = rearrange(
            torch.stack(torch.meshgrid([torch.arange(t), torch.arange(h), torch.arange(h)], indexing="ij")),
            "ndim time height width -> (time height width) ndim",
        ).float().to(self.device)

        dims = torch.tensor([t, h, w]).to(self.device)
        output_pos = output_pos / (dims - 1) * 1000
        
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
            _pred = self.model(_x, output_pos=repeat(output_pos, "... -> b ...", b=b))
            _loss = self.criterion(_pred, _x)
            
        self.scaler.scale(_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        if (self.scheduler is not None) and self.scheduler_step_on_batch:
            self.scheduler.step()
        
        return {"loss": _loss.item()}
    
    @torch.no_grad()
    def _eval_step(self, _x: Tensor, _y: Tensor) -> dict[str, float]:
        b, t, c, h, w = _x.shape
        
        output_pos = rearrange(
            torch.stack(torch.meshgrid([torch.arange(t), torch.arange(h), torch.arange(h)], indexing="ij")),
            "ndim time height width -> (time height width) ndim",
        ).float().to(self.device)

        dims = torch.tensor([t, h, w]).to(self.device)
        output_pos = output_pos / (dims - 1) * 1000
        
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
            _pred = self.eval_model(_x, output_pos=repeat(output_pos, "... -> b ...", b=b))
        
        if self.metrics is not None:
            self.metrics.update(_pred, _x)
        
        return {}
    
    @staticmethod
    def _log_eval(epoch: int, step: int, eval_outs: dict) -> None:
        wandb.log(eval_outs, step=step)
        
        
class ContinuousVideoAutoEncoderUPTEngine(ModelEngine):   
    def __init__(self, model: nn.Module, run_object: DictConfig, resume: bool = False) -> None:
        super().__init__(model, run_object, resume=resume)
        self.metrics = VariableResVideoAutoEncoderMetricCollectionWrapper(
            self.metrics, 
            run_object.dataset.max_rescale_factor
        ) if self.metrics is not None else None
    
    def _train_step(self, _x: Tensor, _y: dict[str, Tensor]) -> dict[str, float]:
        self.model.decoder.unbatch_mode = "none"
        self.optimizer.zero_grad()
        
        coords, values = _y["coords"], _y["values"]
        _x, coords, values = _x.to(self.device), coords.to(self.device), values.to(self.device)
        
        b, t, c, h, w = _x.shape
        
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
            _pred = self.model(_x.to(self.device), output_pos=coords.to(self.device))
            _loss = self.criterion(_pred, values.to(self.device))
            
        self.scaler.scale(_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        if (self.scheduler is not None) and self.scheduler_step_on_batch:
            self.scheduler.step()
        
        return {"loss": _loss.item()}
    
    @torch.no_grad()
    def _eval_step(self, _x: Tensor, _y: Tensor) -> dict[str, float]:
        self.model.decoder.unbatch_mode = "video"
        _x, _y = _x.to(self.device), _y.to(self.device)
        b, x_t, c, x_h, x_w = _x.shape
        b, y_t, c, y_h, y_w = _y.shape
        
        for rescale_factor in range(1, y_h // x_h + 1):
            if x_h * rescale_factor == y_h:
                y = _y.clone()
            else:
                y = F.resize(_y.clone(), (x_h * rescale_factor, x_w * rescale_factor), interpolation=F.InterpolationMode.BILINEAR)
            
            F.normalize(y, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
            
            b, t, c, h, w = y.shape
            
            output_pos = rearrange(
                torch.stack(torch.meshgrid([torch.arange(t), torch.arange(h), torch.arange(w)], indexing="ij")),
                "ndim time height width -> (time height width) ndim",
            ).float().to(self.device)

            dims = torch.tensor([t, h, w]).to(self.device)
            output_pos = output_pos / (dims - 1) * 1000

            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
                _pred = self.eval_model(_x, output_pos=repeat(output_pos, "... -> b ...", b=b))
        
            if self.metrics is not None:
                self.metrics.update(_pred, y, rescale_factor)
        
        return {}
    
    @staticmethod
    def _log_eval(epoch: int, step: int, eval_outs: dict) -> None:
        wandb.log(eval_outs, step=step)
    

class NextFrameUPTEngine(ModelEngine):   
    def __init__(self, model: nn.Module, run_object: DictConfig, resume: bool = False) -> None:
        super().__init__(model, run_object, resume=resume)
        self.metrics = RolloutMetricCollectionWrapper(self.metrics) if self.metrics is not None else None
    
    def _train_step(self, context_frames: Tensor, next_frame: Tensor) -> dict[str, float]:
        self.optimizer.zero_grad()
        
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
            x_next_pred, x_next = self.model.forward_train(context_frames, next_frame)
            loss = self.criterion(x_next_pred, x_next)
            
        self.scaler.scale(loss).backward()           
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        if (self.scheduler is not None) and self.scheduler_step_on_batch:
            self.scheduler.step()
        
        return {"loss": loss.item()}
    
    @torch.no_grad()
    def _eval_step(self, x: Tensor, y: Tensor) -> dict[str, float]:
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
            frames = self.eval_model.rollout_frames(
                x,
                y.shape[1],
            )
            
        if self.metrics is not None:
            self.metrics.update(frames, y)
        
        return {}
    
    @staticmethod
    def _log_eval(epoch: int, step: int, eval_outs: dict) -> None:
        wandb.log(eval_outs, step=step)