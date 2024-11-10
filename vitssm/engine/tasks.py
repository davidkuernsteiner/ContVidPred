from typing import Any, Union

import torch
import numpy as np
from omegaconf.dictconfig import DictConfig
from torch import Tensor, nn
from einops import rearrange

from wandb.sdk.wandb_run import Run
import wandb

from . import ModelEngine
from ..models import UncondUNetModel, NextFrameUNetModel, NextFrameDiTModel
from ..models.vae import frange_cycle_linear
from ..utils.metrics import RolloutMetricCollectionWrapper
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from latte.utils import clip_grad_norm_	
    
    
class VAEEngine(ModelEngine):
    
    def __init__(self, model: AutoencoderKL, run_object: DictConfig) -> None:
        super().__init__(model, run_object)
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
    
    
class DiTNextFrameEngine(ModelEngine):
    
    def __init__(self, model: NextFrameDiTModel, run_object: DictConfig) -> None:
        super().__init__(model, run_object)
        self.metrics = RolloutMetricCollectionWrapper(self.metrics) if self.metrics is not None else None
    
    def _train_step(self, _context_frames: Tensor, _next_frame: Tensor) -> dict[str, float]:
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
    
    
class UncondUNetEngine(ModelEngine):
    
    def __init__(self, model: UncondUNetModel, run_object: DictConfig) -> None:
        super().__init__(model, run_object)
    
    def _train_step(self, _x: Tensor, _y: Tensor) -> dict[str, float]:
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
            _loss = self.model.forward_train(_x)
            
        self.scaler.scale(_loss).backward()           
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        if (self.scheduler is not None) and self.scheduler_step_on_batch:
            self.scheduler.step()
        
        return {"loss": _loss.item()}
    
    @torch.no_grad()
    def _eval_step(self, _x: Tensor, _y: Tensor) -> dict[str, float]:
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
            pass
        #self.metrics.update(_frames, _y)
        
        return {}
    
    @staticmethod
    def _log_eval(epoch: int, step: int, eval_outs: dict) -> None:
        wandb.log(eval_outs, step=step)
        

class UNetNextFrameEngine(ModelEngine):
    
    def __init__(self, model: NextFrameUNetModel, run_object: DictConfig) -> None:
        super().__init__(model, run_object)
        self.metrics = RolloutMetricCollectionWrapper(self.metrics) if self.metrics is not None else None
    
    def _train_step(self, _context_frames: Tensor, _next_frame: Tensor) -> dict[str, float]:
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

class UPTNextFrameEngine(ModelEngine):
        
        def __init__(self, model: nn.Module, run_object: DictConfig) -> None:
            super().__init__(model, run_object)
        
        def _train_step(self, _x: Tensor, _y: Tensor) -> dict[str, float]:
            self.optimizer.zero_grad()
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
                _pred = self.model(_x)
                _loss = self.criterion(_pred, _y)
            self.scaler.scale(_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            if (self.scheduler is not None) and self.scheduler_step_on_batch:
                self.scheduler.step()
            
            return {"loss": _loss.item()}
        
        @torch.no_grad()
        def _eval_step(self, _x: Tensor, _y: Tensor) -> dict[str, float]:
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
                _pred = self.eval_model(_x)
                _loss = self.criterion(_pred, _y)
            
            if self.metrics is not None:
                self.metrics.update(_pred, _y)
            
            return {"loss": _loss.item()}