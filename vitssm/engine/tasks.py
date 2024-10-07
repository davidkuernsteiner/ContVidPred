from typing import Any, Union

import torch
from omegaconf.dictconfig import DictConfig
from torch import Tensor, nn
from einops import rearrange

from wandb.sdk.wandb_run import Run

from . import ModelEngine
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL


class ActionRecognitionEngine(ModelEngine):

    def __init__(self, model: nn.Module, run_object: DictConfig) -> None:
        super().__init__(model, run_object)

    def _train_step(self, _x: Tensor, _y: Tensor) -> tuple[float, dict[str, float]]:
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
            _pred = self.model(_x)
            _loss = self.criterion(_pred, _y)
        self.scaler.scale(_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

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
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
            _pred = self.model(_x)
            _loss = self.criterion(_pred, _y)
        self.scaler.scale(_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

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
    
    
class VideoVAEEngine(ModelEngine):
    
    def __init__(self, model: AutoencoderKL, run_object: DictConfig) -> None:
        super().__init__(model, run_object)
    
    def _train_step(self, _x: Tensor, _y: Tensor) -> tuple[float, dict[str, float]]:
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
            _x = rearrange(_x, "b t c h w -> (b t) c h w")
            _posterior = self.model.encode(_x).latent_dist
            _recon = self.model.decode(_posterior.mode()).sample
            
            beta = min(1.0, self.state["epoch"] / self.config.optimization.epochs)
            _recon_loss = self.criterion(_recon, _x)
            _kl_loss = _posterior.kl().mean()
            _loss = _recon_loss + beta * _kl_loss
            
        self.scaler.scale(_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        
        if (self.scheduler is not None) and self.scheduler_step_on_batch:
            self.scheduler.step()
        
        return _loss.item(), {"recon_loss": _recon_loss.item(), "kl_loss": _kl_loss.item()}
    
    @torch.no_grad()
    def _eval_step(self, _x: Tensor, _y: Tensor) -> tuple[float, Tensor]:
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
            _x = rearrange(_x, "b t c h w -> (b t) c h w")
            _posterior = self.eval_model.encode(_x).latent_dist
            _recon = self.eval_model.decode(_posterior.mode()).sample  
            _loss = self.criterion(_recon, _x) + _posterior.kl().mean()
        
        if self.metrics is not None:
            self.metrics.update(_recon, _x)
        
        return _loss.item(), _recon
