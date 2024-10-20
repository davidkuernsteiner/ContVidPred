from omegaconf import DictConfig
from torch.nn import Module

from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

from .vae import VideoVAEConfig
from .latte import LatteDiffusionModelConfig, LatteDiffusionModel
from .dit import NextFrameDiTModelConfig, NextFrameDiTModel


def build_model(config: DictConfig) -> Module:
    """Builds model given config."""
    match config.model.name:
        case "video-vae":
            model_config = VideoVAEConfig(**config.model)
            model = AutoencoderKL(**model_config.model_dump())
            return model
        
        case "latte":
            model_config = LatteDiffusionModelConfig(**config.model)
            model = LatteDiffusionModel(**model_config.model_dump())
            return model
        
        case "dit":
            model_config = NextFrameDiTModelConfig(**config.model)
            model = NextFrameDiTModel(**model_config.model_dump())
            return model

        case _:
            raise ValueError(f"Model {config.model.name} not supported.")
