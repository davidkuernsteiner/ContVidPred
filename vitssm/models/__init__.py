from omegaconf import DictConfig
from torch.nn import Module
from torchvision.models.video import mvit_v2_s


def build_model(config: DictConfig) -> Module:
    """Builds model given config."""
    model = mvit_v2_s()
    return model
