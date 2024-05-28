from typing import Union, Tuple, Any
from addict import Dict

from torch.utils.data import DataLoader, random_split
from torchvision.transforms import v2 as transforms
import torch


def build_loss(
        config: Dict,
) -> torch.nn.Module:
    """Builds loss function."""
    loss = getattr(torch.nn, config.optimization.loss.name)(**config.optimization.loss.args)
    return loss


def build_optimizer(
        model, config: Dict,
) -> torch.optim.Optimizer:
    """Builds optimizer."""
    optimizer = getattr(torch.optim, config.optimization.optimizer.name)(
        model.parameters(),
        **config.optimization.optimizer.args
    )
    return optimizer


def build_scheduler(
        optimizer,
        config: Dict
) -> torch.optim.lr_scheduler.LRScheduler:
    """Builds learning rate scheduler."""
    if config.optimization.scheduler.name == "LambdaLR":
        warmup_steps = config.optimization.scheduler.args.get("warmup_steps", 10000)
        decay_steps = config.optimization.scheduler.args.get("decay_steps", 100000)
        scheduler_gamma = config.optimization.scheduler.args.get("gamma", 0.5)

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
        **config.optimization.scheduler.args
    )
    return _scheduler
