import gc
import os
from datetime import datetime
from typing import Union

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
import wandb
from wandb import Run
from omegaconf.dictconfig import DictConfig
from torch import Tensor, nn
from tqdm import tqdm

from ..utils import set_seeds
from ..utils.metrics import get_metric_collection

wandb.login()


class ModelEngine:

    def __init__(self, model: nn.Module, run_object: Union[DictConfig, Run]) -> None:
        super().__init__()
        
        if isinstance(run_object, Run):
            self.run = run_object
            self.config = DictConfig(self.run.config)
            
        elif isinstance(run_object, DictConfig):
            self.config = run_object
            self.run = wandb.init(
            config=dict(self.config),
            project=self.config.experiment.project,
            group=self.config.experiment.group,
            name=self.config.experiment.name,
            id=self.config.experiment.name + "_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
            resume="never",
        )
            
        else:
            raise ValueError("Invalid run_object type. Must be a DictConfig or a Run object.")
        
        self.seed = self.config.experiment.get("seed", 42)
        set_seeds(self.seed)

        self.device = torch.device(self.config.model.get("device", "cpu"))
        self.use_amp = self.config.model.get("use_amp", False)

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
        self.metrics = get_metric_collection(self.config)

        self.state = {"step": 0, "epoch": 0}

    def train(self, train_dataloader: DataLoader, eval_dataloader: DataLoader) -> None:
        wandb.watch(
            self.model,
            self.criterion,
            log="all",
            log_freq=self.config.experiment.get("log_freq", 0),
        )

        done = False
        while not done:
            self.state["epoch"] += 1

            self.model.train()
            for x, y in tqdm(train_dataloader, total=len(train_dataloader), desc=f"Epoch {self.state["epoch"]}"):
                loss, _ = self._train_step(x, y)
                self.state["step"] += 1

                if self.state["step"] == self.config.optimization.get("training_steps", 10000):
                    done = True
                    train_metrics = {"loss": loss}
                    self._log_train(self.state["epoch"], self.state["step"], train_metrics)
                    break

                elif self.state["step"] % self.config.experiment.get("log_freq", 0) == 0:
                    train_metrics = {"loss": loss, "learning_rate": self.scheduler.get_last_lr()[-1]}
                    self._log_train(self.state["epoch"], self.state["step"], train_metrics)

            self.model.eval()
            for x, y in eval_dataloader:
                _, _ = self._eval_step(x, y)

            self._log_eval(self.state["epoch"], self.state["step"], self.metrics.compute())

            if (self.scheduler is not None) and (not self.scheduler_step_on_batch):
                self.scheduler.step()

            self._save_checkpoint()
            gc.collect()

    def eval(self) -> None:
        pass

    def load_run(self) -> None:
        self.run = wandb.init(
            config=dict(self.config),
            project=self.config.experiment.wandb.project,
            id=self.config.experiment.wandb.id,
            resume="must",
        )

        self._resume_checkpoint()

    def _train_step(self, _x: Tensor, _y: Tensor) -> tuple[float, Tensor]:
        raise NotImplementedError

    def _eval_step(self, _x: Tensor, _y: Tensor) -> tuple[float, Tensor]:
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
            self.run.project,
            self.run.group,
        )
        os.makedirs(save_dir, exist_ok=True)

        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "state": self.state,
        }
        checkpoint_path = os.path.join(save_dir, self.run.name + ".pth")
        torch.save(checkpoint, checkpoint_path)
        self.run.link_model(path=checkpoint_path, registered_model_name=self.run.name)

    def _resume_checkpoint(self) -> None:
        checkpoint_path = os.path.join(
            os.environ["CHECKPOINT_DIR"],
            self.run.project,
            self.run.group,
            self.run.name + ".pth",
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
    loss = getattr(torch.nn, config.optimization.loss.name)(**config.optimization.loss.args)
    return loss


def get_optimizer(
    model: nn.Module,
    config: DictConfig,
) -> Optimizer:
    """Builds optimizer."""
    optimizer = getattr(torch.optim, config.optimization.optimizer.name)(
        model.parameters(), **config.optimization.optimizer.kwargs
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

        _scheduler = getattr(torch.optim.lr_scheduler, "LambdaLR")(optimizer, lr_lambda=warm_and_decay_lr_scheduler)
        return _scheduler

    _scheduler = getattr(torch.optim.lr_scheduler, config.optimization.scheduler.name)(
        optimizer, **config.optimization.scheduler.kwargs
    )
    return _scheduler
