from torchmetrics import Metric
import torchmetrics
from omegaconf.dictconfig import DictConfig
from torch import Tensor
from torchmetrics import MetricCollection


def build_metric_container(config: DictConfig) -> MetricCollection:
    metrics = {name: getattr(torchmetrics, metric.name)(**metric.kwargs) for name, metric in config.metrics.items()}
    return MetricCollection(metrics)
