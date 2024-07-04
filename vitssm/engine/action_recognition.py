import os
from typing import Tuple

import numpy as np
import torch
from pandas import Series
from torch import nn
from omegaconf.dictconfig import DictConfig
import wandb
from datetime import datetime

from tqdm import tqdm
import gc

from . import ModelEngine, build_loss


class ActionRecognitionEngine(ModelEngine):

    def __init__(self, model: nn.Module, config: DictConfig):
        super().__init__(model, config)

    def train(self, train_loader, eval_loader):
        self.run = wandb.init(
            config=dict(self.config),
            project=self.config.experiment.wandb.project,
            group=self.config.experiment.wandb.group,
            name=self.config.experiment.wandb.name + "_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
        )
        wandb.watch(
            self.model,
            self.criterion, log="all",
            log_freq=self.config.experiment.get("log_freq", 0),
        )

        step = 0
        done = False
        epoch = 0
        while not done:
            epoch += 1

            self.model.train()
            for x, y in tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch}"):
                loss, _ = self._train_step(x, y)
                step += 1

                if step == self.config.optimization.get("training_steps", 10000):
                    done = True
                    train_metrics = {"loss": loss}
                    self._log_train(epoch, step, train_metrics)
                    break

                elif step % self.config.experiment.get("log_freq", 0) == 0:
                    train_metrics = {"loss": loss, "learning_rate": self.scheduler.get_last_lr()[-1]}
                    self._log_train(epoch, step, train_metrics)

            self.model.eval()
            for x, y in eval_loader:
                batch_eval_metrics, preds = self._eval_step(x, y)
                eval_metrics += batch_eval_metrics
                x = x
                y = y

            self._log_val(epoch, step, self.metrics.compute())

            if (self.scheduler is not None) and (not self.scheduler_step_on_batch):
                self.scheduler.step()

            self._save_checkpoint()
            gc.collect()

        self.run.finish()

    def _train_step(self, _x, _y) -> Tuple[float, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        _x, _y = _x.to(self.device), _y.to(self.device)

        _pred = self.model(_x)

        _loss = self.criterion(_pred, _x)
        self.optimizer.zero_grad()
        _loss.backward()
        self.optimizer.step()
        if (self.scheduler is not None) and self.scheduler_step_on_batch:
            self.scheduler.step()

        return _loss.item(), _pred

    @torch.no_grad()
    def _eval_step(self, _x, _y) -> Tuple[Series, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        _x, _y = _x.to(self.device), _y.to(self.device)

        _pred = self.model(_x)

        _loss = self.criterion(_recon_combined, _x)
        _batch_fg_ari, _batch_ari = preprocess_and_batch_aris(_y, _soft_pred_masks)

        _metrics = Series({
            "eval_loss": _loss.item(),
            "eval_ari_fg": _batch_fg_ari.mean().item(),
            "eval_ari": _batch_ari.mean().item()
            }
        )

        return _metrics, (_recon_combined, _recons, _soft_pred_masks, _slots)

    @staticmethod
    def _log_train(epoch: int, step: int, train_metrics: dict) -> None:
        wandb.log({"train": train_metrics, "epoch": epoch}, step=step)

    @staticmethod
    def _log_val(epoch: int, step: int, eval_metrics: dict) -> None:
        wandb.log({"eval": eval_metrics, "epoch": epoch}, step=step)

    def _save_checkpoint(self) -> None:
        save_dir = os.path.join(self.config.experiment.get("checkpoint_path", "checkpoints"),
                                 self.run.project,
                                 self.run.group
                                )
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(save_dir, self.run.name + ".pth"))
