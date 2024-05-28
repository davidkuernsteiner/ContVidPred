from abc import ABC, abstractmethod

import torch
import wandb
from addict import Dict
from torch import nn

from .constructors import build_optimizer, build_scheduler, build_loss
from .next_frame_prediction import NextFrameEngine
from ..utils import set_seeds

wandb.login()


class ModelEngine(ABC):

    def __init__(self, model: nn.Module, config: Dict):
        self.config = config
        self.seed = config.experiment.get("seed", 42)
        self.device = torch.device(config.model.get("device", "cpu"))
        self.model = model.to(self.device)
        self.optimizer = build_optimizer(model, config)
        if config.optimization.get("scheduler", None) is not None:
            self.scheduler = build_scheduler(self.optimizer, config)
            self.scheduler_step_on_batch = config.optimization.scheduler.get("step_on_batch", False)
        else:
            self.scheduler = None
            self.scheduler_step_on_batch = False

        self.run = None
        set_seeds(self.seed)

    @abstractmethod
    def train(self, train_loader, val_loader):
        pass

    @abstractmethod
    def _train_step(self, _x, _y) -> float:
        pass

    @abstractmethod
    def _eval_step(self, _x, _y) -> float:
        pass

    @abstractmethod
    def _save_checkpoint(self) -> None:
        pass