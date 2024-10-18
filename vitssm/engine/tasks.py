from typing import Any, Union

import torch
from omegaconf.dictconfig import DictConfig
from torch import Tensor, nn
from einops import rearrange

from wandb.sdk.wandb_run import Run

from . import ModelEngine
from ..models import LatteDiffusionModel
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from latte.utils import clip_grad_norm_	


class ActionRecognitionEngine(ModelEngine):

    def __init__(self, model: nn.Module, run_object: DictConfig) -> None:
        super().__init__(model, run_object)

    def _train_step(self, _x: Tensor, _y: Tensor) -> tuple[float, dict[str, float]]:
        self.optimizer.zero_grad()
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
            _pred = self.model(_x)
            _loss = self.criterion(_pred, _y)
        self.scaler.scale(_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if (self.scheduler is not None) and self.scheduler_step_on_batch:
            self.scheduler.step()

        return _loss.item(), _pred

    @torch.no_grad()
    def _eval_step(self, _x: Tensor, _y: Tensor) -> tuple[float, Tensor]:
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
            _pred = self.eval_model(_x)
            _loss = self.criterion(_pred, _y)

        self.metrics.update(_pred, _y)

        return _loss.item(), _pred


class NextFrameEngine(ModelEngine):

    def __init__(self, model: nn.Module, run_object: DictConfig) -> None:
        super().__init__(model, run_object)

    def _train_step(self, _x: Tensor, _y: Tensor) -> tuple[float, dict[str, float]]:
        self.optimizer.zero_grad()
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
            _pred = self.model(_x)
            _loss = self.criterion(_pred, _y)
        self.scaler.scale(_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if (self.scheduler is not None) and self.scheduler_step_on_batch:
            self.scheduler.step()

        return _loss.item(), _pred

    @torch.no_grad()
    def _eval_step(self, _x: Tensor, _y: Tensor) -> tuple[float, Tensor]:
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
            _pred = self.eval_model(_x)
            _loss = self.criterion(_pred, _y)

        if self.metrics is not None:
            self.metrics.update(_pred, _y)

        return _loss.item(), _pred
    
    
class VideoVAEEngine(ModelEngine):
    
    def __init__(self, model: AutoencoderKL, run_object: DictConfig) -> None:
        super().__init__(model, run_object)
    
    def _train_step(self, _x: Tensor, _y: Tensor) -> dict[str, float]:
        self.optimizer.zero_grad()
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
            _x = rearrange(_x, "b t c h w -> (b t) c h w")
            _posterior = self.model.encode(_x).latent_dist
            _recon = self.model.decode(_posterior.sample()).sample
            
            beta = 1e-5 #min(1.0, self.state["epoch"] / (self.config.optimization.epochs * 0.5))
            _recon_loss = self.criterion(_recon, _x)
            _kl_loss = _posterior.kl().mean()
            _loss = _recon_loss + beta * _kl_loss
            
        self.scaler.scale(_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        if (self.scheduler is not None) and self.scheduler_step_on_batch:
            self.scheduler.step()
        
        return {"loss": _loss.item(), "recon_loss": _recon_loss.item(), "kl_loss": _kl_loss.item()}
    
    @torch.no_grad()
    def _eval_step(self, _x: Tensor, _y: Tensor) -> dict[str, float]:
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
            _x = rearrange(_x, "b t c h w -> (b t) c h w")
            _posterior = self.eval_model.encode(_x).latent_dist
            _recon = self.eval_model.decode(_posterior.sample()).sample
            
            beta = min(1.0, self.state["epoch"] / self.config.optimization.epochs)
            _recon_loss = self.criterion(_recon, _x)
            _kl_loss = _posterior.kl().mean()
            _loss = _recon_loss + beta * _kl_loss
        
        if self.metrics is not None:
            self.metrics.update(_recon, _x)
        
        return {"loss": _loss.item(), "recon_loss": _recon_loss.item(), "kl_loss": _kl_loss.item()}
    
    
class LatteEngine(ModelEngine):
    
    def __init__(self, model: LatteDiffusionModel, run_object: DictConfig) -> None:
        super().__init__(model, run_object)
    
    def _train_step(self, _x: Tensor, _y: Tensor) -> dict[str, float]:
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
            _loss = self.model.forward_train(_x)
            
        self.scaler.scale(_loss).backward()
        
        if self.state["step"] < self.model.start_clip_iter: # if train_steps >= start_clip_iter, will clip gradient
            _gradient_norm = clip_grad_norm_(self.model.latte.module.parameters(), self.model.clip_max_norm, clip_grad=False)
        else:
            _gradient_norm = clip_grad_norm_(self.model.latte.module.parameters(), self.model.clip_max_norm, clip_grad=True)
            
        if (self.state["step"] % self.model.gradient_accumulation_steps == 0) and (self.state["step"] > 0):
            self.scaler.step(self.optimizer)
            self.scaler.update()
        
        if (self.scheduler is not None) and self.scheduler_step_on_batch:
            self.scheduler.step()
        
        return {"loss": _loss.item(), "grad_norm": _gradient_norm.item()}
    
    @torch.no_grad()
    def _eval_step(self, _x: Tensor, _y: Tensor) -> dict[str, float]:
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
            _loss = self.model.forward_train(_x)
        
        return {"loss": _loss.item()}