from typing import Any, Union
import torch
from omegaconf.dictconfig import DictConfig
from wandb.sdk.wandb_run import Run
from torch import Tensor, nn

from . import ModelEngine


class ActionRecognitionEngine(ModelEngine):

    def __init__(self, model: nn.Module, run_object: Union[Run, DictConfig]):
        super().__init__(model, run_object)

    def _train_step(self, _x: Tensor, _y: Tensor) -> tuple[float, Tensor]:
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
            _pred = self.model(_x)
            _loss = self.criterion(_pred, _y)

        self.metrics.update(_pred, _y)

        return _loss.item(), _pred


class NextFrameEngine(ModelEngine):

    def __init__(self, model: nn.Module, run_object: Union[Run, DictConfig]):
        super().__init__(model, run_object)

    def _train_step(self, _x: Tensor, _y: Tensor) -> tuple[float, Tensor]:
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
            _pred = self.model(_x)
            _loss = self.criterion(_pred, _y)

        self.metrics.update(_pred, _y)

        return _loss.item(), _pred
