import torchmetrics
from omegaconf.dictconfig import DictConfig
from torchmetrics import MetricCollection


def get_metric_collection(config: DictConfig) -> MetricCollection:
    metrics = {name: getattr(torchmetrics, metric.name)(**metric.get("kwargs", {})) for name, metric in config.metrics.items()}
    return MetricCollection(metrics)
