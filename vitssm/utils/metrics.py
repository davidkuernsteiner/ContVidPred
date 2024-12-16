from collections import defaultdict
from typing import Sequence, Union
import numpy as np
from torch import Tensor
from einops import rearrange
from pandas import DataFrame
import torchmetrics
from omegaconf.dictconfig import DictConfig
import random
from torchmetrics import Metric, MetricCollection
import torchmetrics.image
import wandb
from copy import deepcopy

from .visual import model_output_to_image, model_output_to_video


def get_metric_collection(config: DictConfig) -> MetricCollection:
    metrics = {name: getattr(torchmetrics.image, metric.name)(**metric.get("kwargs", {})) for name, metric in config.metrics.items()}
    return MetricCollection(metrics)


class AutoEncoderMetricCollectionWrapper:
    def __init__(self, metrics: MetricCollection) -> None:
        """Calculate metrics over the T dimension of [N, T, ...] tensors."""
        super().__init__()
        self.metrics = metrics
        self.sample = np.zeros(0)
        self.sample_pred = np.zeros(0)
        
    def update(self, x: Tensor, y: Tensor) -> None:
        sample_idx = random.randint(0, x.size(0) - 1)
        self.sample_pred = model_output_to_image(x.clone()[sample_idx])
        self.sample = model_output_to_image(y.clone()[sample_idx])
        self.metrics.update(x, y)
        
    def compute(self) -> dict:
        samples = {
            "ground truth vs. prediction": [
                wandb.Image(self.sample),
                wandb.Image(self.sample_pred),
            ],
        }
            
        return self.metrics.compute() | samples
    
    def reset(self) -> None:
        self.metrics.reset()
        self.sample = np.zeros(0)
        self.sample_pred = np.zeros(0)


class VideoAutoEncoderMetricCollectionWrapper:
    def __init__(self, metrics: MetricCollection) -> None:
        """Calculate metrics over the T dimension of [N, T, ...] tensors."""
        super().__init__()
        self.metrics = metrics
        self.sample = np.zeros(0)
        self.sample_pred = np.zeros(0)
        
    def update(self, x: Tensor, y: Tensor) -> None:
        sample_idx = random.randint(0, x.size(0) - 1)
        self.sample_pred = model_output_to_video(x.clone()[sample_idx])
        self.sample = model_output_to_video(y.clone()[sample_idx])
        x = rearrange(x, "b t c h w -> (b t) c h w")
        y = rearrange(y, "b t c h w -> (b t) c h w")
        self.metrics.update(x, y)
        
    def compute(self) -> dict:
        samples = {
            "ground truth vs. prediction": [
                wandb.Video(self.sample,fps=4),
                wandb.Video(self.sample_pred, fps=4),
            ],
        }
            
        return self.metrics.compute() | samples
    
    def reset(self) -> None:
        self.metrics.reset()
        self.sample = np.zeros(0)
        self.sample_pred = np.zeros(0)
        

class VariableResolutionVideoAutoEncoderMetricCollectionWrapper:
    def __init__(self, metrics: MetricCollection, max_rescale_factor: int) -> None:
        """Calculate metrics over the T dimension of [N, T, ...] tensors."""
        super().__init__()
        self.metrics = [deepcopy(metrics) for _ in range(max_rescale_factor)]
        self.samples = [np.zeros(0) for _ in range(max_rescale_factor)]
        self.samples_pred = [np.zeros(0) for _ in range(max_rescale_factor)]
        
    def update(self, x: Tensor, y: Tensor, rescale_factor: int) -> None:
        sample_idx = random.randint(0, x.size(0) - 1)
        self.samples_pred[rescale_factor - 1] = model_output_to_video(x.clone()[sample_idx])
        self.samples[rescale_factor - 1] = model_output_to_video(y.clone()[sample_idx])
        x = rearrange(x, "b t c h w -> (b t) c h w")
        y = rearrange(y, "b t c h w -> (b t) c h w")
        self.metrics[rescale_factor - 1].update(x, y)
        
    def compute(self) -> dict:
        outs = {}
        for i in range(len(self.metrics)):
            metrics = self.metrics[i].compute()
            samples = {
                "ground truth vs. prediction": [
                    wandb.Video(self.samples[i],fps=4),
                    wandb.Video(self.samples_pred[i], fps=4),
                ],
            }
            
            outs[f"rescale_factor_{i + 1}"] = metrics | samples
            
        return outs
    
    def reset(self) -> None:
        for i in range(len(self.metrics)):
            self.metrics[i].reset()
            self.samples[i] = np.zeros(0)
            self.samples_pred[i] = np.zeros(0)
        

class RolloutMetricCollectionWrapper:
    def __init__(self, metrics: MetricCollection) -> None:
        """Calculate metrics over the T dimension of [N, T, ...] tensors."""
        super().__init__()
        self.metrics = metrics
        self.results = []
        self.sample_frames = np.zeros(0)
        self.sample_frames_pred = np.zeros(0)
        
    def update(self, x: Tensor, y: Tensor) -> None:
        sample_idx = random.randint(0, x.size(0) - 1)
        self.sample_frames_pred = model_output_to_video(x.clone()[sample_idx])
        self.sample_frames = model_output_to_video(y.clone()[sample_idx])
        x, y = rearrange(x, "n t ... -> t n ..."), rearrange(y, "n t ... -> t n ...")
        res = {timestep + 1: self.metrics(_x, _y) for timestep, (_x, _y) in enumerate(zip(x, y))}
        res = DataFrame(res).T.map(lambda x: x.item())
        self.results.append(res)
        
    def compute(self) -> dict:
        assert len(self.results) > 0, "No results to compute."
        res = (sum(self.results) / len(self.results)).to_dict()
        metrics = {}
        for metric, values in res.items():
            data = [(int(step), value) for step, value in values.items()]
            table = wandb.Table(data=data, columns=["step", metric])
            metrics[metric] = wandb.plot.line(table, x="step", y=metric, title=f"{metric} over rollout steps")
        
        sample_frames = {
            "rollout: ground truth vs. prediction": [
                wandb.Video(self.sample_frames,fps=4),
                wandb.Video(self.sample_frames_pred, fps=4),
            ],
        }
            
        return metrics | sample_frames
    
    def reset(self) -> None:
        self.metrics.reset()
        self.results = []
        self.sample_frames = np.zeros(0)
        self.sample_frames_pred = np.zeros(0)


class VariableResolutionRolloutMetricCollectionWrapper:
    def __init__(self, metrics: MetricCollection, max_rescale_factor: int) -> None:
        """Calculate metrics over the T dimension of [N, T, ...] tensors."""
        super().__init__()
        self.metrics = metrics
        self.results = [[]] * max_rescale_factor
        self.samples = [np.zeros(0) for _ in range(max_rescale_factor)]
        self.samples_pred = [np.zeros(0) for _ in range(max_rescale_factor)]
        self.max_rescale_factor = max_rescale_factor
        
    def update(self, x: Tensor, y: Tensor, rescale_factor: int) -> None:
        sample_idx = random.randint(0, x.size(0) - 1)
        self.samples_pred[rescale_factor - 1] = model_output_to_video(x.clone()[sample_idx])
        self.samples[rescale_factor - 1] = model_output_to_video(y.clone()[sample_idx])
        x, y = rearrange(x, "n t ... -> t n ..."), rearrange(y, "n t ... -> t n ...")
        res = {timestep + 1: self.metrics(_x, _y) for timestep, (_x, _y) in enumerate(zip(x, y))}
        res = DataFrame(res).T.map(lambda x: x.item())
        self.results[rescale_factor - 1].append(res)
        
    def compute(self) -> dict:
        results = {}
        metrics_y = defaultdict(list)
        for i in range(len(self.results)):
            assert len(self.results[i]) > 0, "No results to compute."
            res = (sum(self.results[i]) / len(self.results[i])).to_dict()
            for metric, values in res.items():
                metrics_y[metric].append(values)

            sample_frames = {
                "rollout: ground truth vs. prediction": [
                    wandb.Video(self.samples[i],fps=4),
                    wandb.Video(self.samples_pred[i], fps=4),
                ],
            }

            results[f"rescale_factor_{i + 1}"] = sample_frames
        
        for metric, values in metrics_y.items():
            results[metric] = wandb.plot.line_series(
                xs=list(range(1, len(values) + 1)), 
                ys=values,
                keys=[f"scale: {i + 1}" for i in range(len(values))],
                title=f"{metric} over rollout steps",
                xname="rollout step",
            )
            
        return results
    
    def reset(self) -> None:
        for i in range(len(self.metrics)):
            self.metrics.reset()
            self.samples[i] = np.zeros(0)
            self.samples_pred[i] = np.zeros(0)
            self.results = [[]] * self.max_rescale_factor
