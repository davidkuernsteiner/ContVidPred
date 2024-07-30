import torchmetrics
from omegaconf.dictconfig import DictConfig
from torchmetrics import MetricCollection


def build_metric_collection(config: DictConfig) -> MetricCollection:
    metrics = {name: getattr(torchmetrics, metric.name)(**metric.kwargs) for name, metric in config.metrics.items()}
    return MetricCollection(metrics)
