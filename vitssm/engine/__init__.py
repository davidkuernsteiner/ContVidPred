from collections import defaultdict
import gc
import os
from datetime import datetime
from typing import Any, Union

import torch
from omegaconf.dictconfig import DictConfig
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn, update_bn
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from wandb.sdk.wandb_run import Run

from ..utils import count_parameters, set_seeds
from ..utils.metrics import get_metric_collection

wandb.login()


class ModelEngine:
    def __init__(self, model: nn.Module, run_object: DictConfig) -> None:
        super().__init__()

        self.config = run_object
        self.seed = self.config.get("seed", 42)
        set_seeds(self.seed)

        self.device = torch.device(self.config.model.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.use_amp = self.config.model.get("use_amp", True)
        self.use_ema = self.config.model.get("use_ema", True)
        self.ema_steps = self.config.model.get("ema_steps", 1)

        self.model = model.to(self.device)            
        self.optimizer = get_optimizer(model, self.config)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        if self.config.optimization.get("scheduler", None) is not None:
            self.scheduler = get_scheduler(self.optimizer, self.config)
            self.scheduler_step_on_batch = self.config.optimization.scheduler.get("step_on_batch", False)

        else:
            self.scheduler = None
            self.scheduler_step_on_batch = False

        self.criterion = get_loss(self.config)
        self.metrics = get_metric_collection(self.config).to(self.device) if self.config.get("metrics", None) is not None else None
        
        if self.use_ema:
            self.ema = AveragedModel(
                self.model,
                device=self.device,
                multi_avg_fn=get_ema_multi_avg_fn(0.999),
                use_buffers=True
            )
            self.eval_model = self.ema.module
        else:
            self.eval_model = self.model

        self.state = {"step": 0, "epoch": 0}
        wandb.log({"Model Parameters": count_parameters(self.model)})

    def train(self, train_dataloader: DataLoader, eval_dataloader: DataLoader) -> None:
        wandb.watch(
            self.model,
            self.criterion,
            log="all",
            log_freq=self.config.get("log_freq", 0),
        )

        done = False
        while not done:
            self.state["epoch"] += 1

            self.model.train()
            for x, y in tqdm(train_dataloader, total=len(train_dataloader), desc=f"Epoch {self.state["epoch"]}"):
                loss, train_outs = self._train_step(x.to(self.device), y.to(self.device))
                self.state["step"] += 1
                
                if self.use_ema and (self.state["step"] % self.ema_steps == 0):
                    self.ema.update_parameters(self.model)

                if self.state["step"] == self.config.optimization.get("steps", torch.inf):
                    done = True
                    train_metrics = {"loss": loss}
                    self._log_train(self.state["epoch"], self.state["step"], train_metrics)
                    break
                elif self.state["epoch"] == self.config.optimization.get("epochs", torch.inf):
                    done = True

                if self.state["step"] % self.config.get("log_freq", 0) == 0:
                    train_metrics = {"loss": loss, "learning_rate": self.scheduler.get_last_lr()[-1]} | train_outs
                    self._log_train(self.state["epoch"], self.state["step"], train_metrics)

            if self.use_ema:
                update_bn(train_dataloader, self.ema, device=self.device)
                
            
            eval_metrics = defaultdict(list)
            self.eval_model.eval()
            for x, y in eval_dataloader:
                eval_outs = self._eval_step(x.to(self.device), y.to(self.device))
                for k, v in eval_outs.items():
                    eval_metrics[k].append(v)
                
            eval_metrics = {k: sum(v) / len(v) for k, v in eval_metrics.items() if v}
            self._log_eval(self.state["epoch"], self.state["step"], self.metrics.compute() | eval_metrics if self.metrics is not None else eval_metrics)

            if (self.scheduler is not None) and (not self.scheduler_step_on_batch):
                self.scheduler.step()

            self._save_checkpoint()
            gc.collect()

    def eval(self) -> None:
        pass

    def load_run(self) -> None:
        self.run = wandb.init(
            config=dict(self.config),
            project=self.config.project,
            id=self.config.id,
            resume="must",
        )

        self._resume_checkpoint()

    def _train_step(self, _x: Tensor, _y: Tensor) -> tuple[float, dict[str, Any]]:
        raise NotImplementedError

    def _eval_step(self, _x: Tensor, _y: Tensor) -> dict[str, float]:
        raise NotImplementedError

    @staticmethod
    def _log_train(epoch: int, step: int, train_metrics: dict) -> None:
        wandb.log({"train": train_metrics, "epoch": epoch}, step=step)

    @staticmethod
    def _log_eval(epoch: int, step: int, eval_metrics: dict) -> None:
        wandb.log({"eval": eval_metrics, "epoch": epoch}, step=step)

    def _save_checkpoint(self) -> None:
        save_dir = os.path.join(
            os.environ["CHECKPOINT_DIR"],
            self.config.project,
            self.config.group,
        )
        os.makedirs(save_dir, exist_ok=True)

        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "state": self.state,
        } | {"ema": self.ema.state_dict()} if self.use_ema else {}
        
        checkpoint_path = os.path.join(save_dir, self.config.name + ".pth")
        torch.save(checkpoint, checkpoint_path)
        wandb.link_model(path=checkpoint_path, registered_model_name=self.config.name)

    def _resume_checkpoint(self) -> None:
        checkpoint_path = os.path.join(
            os.environ["CHECKPOINT_DIR"],
            self.config.project,
            self.config.group,
            self.config.name + ".pth",
        )

        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scaler.load_state_dict(checkpoint["scaler"])
        self.state = checkpoint["state"]

    def _early_stopping_check(self) -> bool:
        raise NotImplementedError


def get_loss(
    config: DictConfig,
) -> torch.nn.Module:
    """Builds loss function."""
    loss = getattr(torch.nn, config.optimization.loss.name)(**config.optimization.loss.get("kwargs", {}))
    return loss


def get_optimizer(
    model: nn.Module,
    config: DictConfig,
) -> Optimizer:
    """Builds optimizer."""
    optimizer = getattr(torch.optim, config.optimization.optimizer.name)(
        model.parameters(), **config.optimization.optimizer.get("kwargs", {}),
    )
    return optimizer


def get_scheduler(optimizer: Optimizer, config: DictConfig) -> LRScheduler:
    """Builds learning rate scheduler."""
    if config.optimization.scheduler.name == "LambdaLR":
        warmup_steps = config.optimization.scheduler.kwargs.get("warmup_steps", 10000)
        decay_steps = config.optimization.scheduler.kwargs.get("decay_steps", 100000)
        scheduler_gamma = config.optimization.scheduler.kwargs.get("gamma", 0.5)

        def warm_and_decay_lr_scheduler(step: int):
            if step < warmup_steps:
                return step / warmup_steps

            return scheduler_gamma ** (step / decay_steps)

        _scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_and_decay_lr_scheduler)
        return _scheduler

    _scheduler = getattr(torch.optim.lr_scheduler, config.optimization.scheduler.name)(
        optimizer, **config.optimization.scheduler.get("kwargs", {}),
    )
    return _scheduler
