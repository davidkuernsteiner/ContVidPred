from torchmetrics import Metric
import torchmetrics
from omegaconf.dictconfig import DictConfig



class MetricContainer:
    def __init__(self, metrics: list[Metric]) -> None:
        self.metrics = metrics

    def __call__(self, *args, **kwargs) -> dict:
        return {metric.name: metric(*args, **kwargs) for metric in self.metrics}

    def update(self, *args, **kwargs) -> None:
        for metric in self.metrics:
            metric.update(*args, **kwargs)

    def compute(self) -> dict:
        return {metric.name: metric.compute() for metric in self.metrics}


def build_metric_container(config: DictConfig) -> MetricContainer:
    metrics = [getattr(torchmetrics, metric.name)(**metric.kwargs) for metric in config.metrics]
    return MetricContainer(metrics)
