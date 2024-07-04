from abc import ABC, abstractmethod

import torch
import wandb
from omegaconf.dictconfig import DictConfig
from torch import nn

from ..utils import set_seeds
from ..utils.metrics import build_metric_container

wandb.login()


class ModelEngine(ABC):

    def __init__(self, model: nn.Module, config: DictConfig) -> None:
        self.config = config
        self.seed = config.experiment.get("seed", 42)
        self.device = torch.device(config.model.get("device", "cpu"))
        self.use_amp = config.model.get("use_amp", False)

        self.model = model.to(self.device)
        self.optimizer = build_optimizer(model, config)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        if config.optimization.get("scheduler", None) is not None:
            self.scheduler = build_scheduler(self.optimizer, config)
            self.scheduler_step_on_batch = config.optimization.scheduler.get("step_on_batch", False)

        else:
            self.scheduler = None
            self.scheduler_step_on_batch = False

        self.criterion = build_loss(config)
        self.metrics = build_metric_container(config)

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

    @abstractmethod
    def _resume_checkpoint(self) -> None:
        pass

    @abstractmethod
    def _early_stopping_check(self) -> bool:
        pass


def build_loss(
        config: DictConfig,
) -> torch.nn.Module:
    """Builds loss function."""
    loss = getattr(torch.nn, config.optimization.loss.name)(**config.optimization.loss.args)
    return loss


def build_optimizer(
        model, config: DictConfig,
) -> torch.optim.Optimizer:
    """Builds optimizer."""
    optimizer = getattr(torch.optim, config.optimization.optimizer.name)(
        model.parameters(),
        **config.optimization.optimizer.kwargs
    )
    return optimizer


def build_scheduler(
        optimizer,
        config: DictConfig
) -> torch.optim.lr_scheduler.LRScheduler:
    """Builds learning rate scheduler."""
    if config.optimization.scheduler.name == "LambdaLR":
        warmup_steps = config.optimization.scheduler.kwargs.get("warmup_steps", 10000)
        decay_steps = config.optimization.scheduler.kwargs.get("decay_steps", 100000)
        scheduler_gamma = config.optimization.scheduler.kwargs.get("gamma", 0.5)

        def warm_and_decay_lr_scheduler(step: int):
            if step < warmup_steps:
                return step / warmup_steps

            return scheduler_gamma ** (step / decay_steps)

        _scheduler = getattr(torch.optim.lr_scheduler, "LambdaLR")(
            optimizer,
            lr_lambda=warm_and_decay_lr_scheduler
        )
        return _scheduler

    _scheduler = getattr(torch.optim.lr_scheduler, config.optimization.scheduler.name)(
        optimizer,
        **config.optimization.scheduler.kwargs
    )
    return _scheduler