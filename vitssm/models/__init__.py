from omegaconf import DictConfig
from torch import nn
from torch.nn import Module
from torch.optim import Optimizer
from torch.cuda.amp import GradScaler
from torchvision.models.video import swin3d_t

def build_model(
        config: DictConfig
) -> Module:
    """Builds model given config."""
    model = swin3d_t()
    return model