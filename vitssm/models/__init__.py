from torch import nn
from torch.nn import Module
from torch.optim import Optimizer
from torch.cuda.amp import GradScaler

class OptimizedModel(Module):

    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        scaler: GradScaler,
        loss: Module,
    ) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler

    def forward(self, x):
        pass
