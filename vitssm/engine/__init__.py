import gc
import os
from datetime import datetime

import torch
import wandb
from omegaconf.dictconfig import DictConfig
from torch import nn
from tqdm import tqdm

from ..data import build_dataloaders
from ..utils import set_seeds
from ..utils.metrics import build_metric_container

wandb.login()


class ModelEngine:

    def __init__(self, model: nn.Module, config: DictConfig) -> None:
        self.config = config
        self.seed = config.experiment.get("seed", 42)
        set_seeds(self.seed)

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

        self.train_loader, self.eval_loader = build_dataloaders(config)

        self.run = None
        self.state = {"step": 0, "epoch": 0}

    def train(self, train_loader, eval_loader):
        self.run = wandb.init(
            config=dict(self.config),
            project=self.config.experiment.wandb.project,
            group=self.config.experiment.wandb.group,
            name=self.config.experiment.wandb.name + "_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
            id=self.config.experiment.wandb.id,
            resume="allow",
        )
        wandb.watch(
            self.model,
            self.criterion, log="all",
            log_freq=self.config.experiment.get("log_freq", 0),
        )

        done = False
        while not done:
            self.state["epoch"] += 1

            self.model.train()
            for x, y in tqdm(train_loader, total=len(train_loader), desc=f"Epoch {self.state["epoch"]}"):
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
            for x, y in eval_loader:
                batch_eval_metrics, preds = self._eval_step(x, y)

            self._log_eval(self.state["epoch"], self.state["step"], self.metrics.compute())

            if (self.scheduler is not None) and (not self.scheduler_step_on_batch):
                self.scheduler.step()

            self._save_checkpoint()
            gc.collect()

        self.run.finish()

    def eval(self) -> None:
        pass

    def load_run(self) -> None:
        self.run = wandb.init(
            config=dict(self.config),
            project=self.config.experiment.wandb.project,
            id = self.config.experiment.wandb.id,
            resume="must",
        )

        self._resume_checkpoint()

    def _train_step(self, _x, _y) -> float:
        raise NotImplementedError

    def _eval_step(self, _x, _y) -> float:
        raise NotImplementedError

    @staticmethod
    def _log_train(epoch: int, step: int, train_metrics: dict) -> None:
        wandb.log({"train": train_metrics, "epoch": epoch}, step=step)

    @staticmethod
    def _log_eval(epoch: int, step: int, eval_metrics: dict) -> None:
        wandb.log({"eval": eval_metrics, "epoch": epoch}, step=step)

    def _save_checkpoint(self) -> None:
        save_dir = os.path.join(
            self.config.experiment.get("checkpoint_path", "checkpoints"),
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
            self.config.experiment.get("checkpoint_path", "checkpoints"),
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