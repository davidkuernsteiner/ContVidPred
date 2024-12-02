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
        
    def compute(self) -> DataFrame:
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
        self.results = []
