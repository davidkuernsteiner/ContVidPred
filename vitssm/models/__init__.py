from omegaconf import DictConfig
from torch.nn import Module

from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

from .vae import VideoVAEConfig
from .latte import LatteDiffusionModelConfig, LatteDiffusionModel
from .dit import NextFrameDiTModelConfig, NextFrameDiTModel


def build_model(config: DictConfig) -> Module:
    """Builds model given config."""
    match config.name:
        case "vae":
            model_config = VideoVAEConfig(**config)
            model = AutoencoderKL(**model_config.model_dump())
            return model
        
        case "latte":
            model_config = LatteDiffusionModelConfig(**config)
            model = LatteDiffusionModel(**model_config.model_dump())
            return model
        
        case "dit":
            model_config = NextFrameDiTModelConfig(**config)
            model = NextFrameDiTModel(**model_config.model_dump())
            return model

        case _:
            raise ValueError(f"Model {config.name} not supported.")
