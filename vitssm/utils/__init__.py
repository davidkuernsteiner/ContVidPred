import random

from PIL import Image
from einops import rearrange
import numpy as np
from torch import nn
import torch
from omegaconf import DictConfig


def set_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def flatten_config(config: DictConfig, parent_key: str = None, sep: str = "_"):
    items = []
    for k, v in config.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, DictConfig):
            items.extend(flatten_config(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return DictConfig(dict(items))


def ema_to_model_state_dict(ema_dict):
    model_dict = {}
    for key, value in ema_dict.items():
        if key.startswith("module."):
            model_dict[key[7:]] = value
    return model_dict


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def min_max_normalize(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor - tensor.min()) / (tensor.max() - tensor.min())


def model_output_to_image(x: torch.Tensor):
    return Image.fromarray(rearrange(min_max_normalize(x).detach().cpu().numpy() * 255, "c h w -> h w c").astype("uint8"))


def model_output_to_video(x: torch.Tensor):
    return rearrange(min_max_normalize(x).detach().cpu() * 255, "t c h w -> t h w c").to(torch.uint8)