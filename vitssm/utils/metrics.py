from torchmetrics import Metric
import torchmetrics
from omegaconf.dictconfig import DictConfig
from torch import Tensor



class MetricContainer:
    def __init__(self, metrics: dict[str, Metric]) -> None:
        self.metrics = metrics

    def __call__(self, pred: Tensor, target: Tensor) -> dict:
        return {name: metric(pred, target) for name, metric in self.metrics.items()}

    def update(self, pred: Tensor, target: Tensor) -> None:
        for metric in self.metrics.values():
            metric.update(pred, target)

    def compute(self) -> dict:
        return {name: metric.compute() for name, metric in self.metrics.items()}

    def reset(self) -> None:
        for metric in self.metrics.values():
            metric.reset()


def build_metric_container(config: DictConfig) -> MetricContainer:
    metrics = {name: getattr(torchmetrics, metric.name)(**metric.kwargs) for name, metric in config.metrics.items()}
    return MetricContainer(metrics)
