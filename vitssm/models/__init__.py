from omegaconf import DictConfig
from torch.nn import Module

from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

from .lnfp import LNFP, LNFPConfig
from .lnfp_simple import LNFPSimple, LNFPSimpleConfig
from .vae import VideoVAEConfig


def build_model(config: DictConfig) -> Module:
    """Builds model given config."""
    match config.model.name:

        case "lnfp":
            model_config = LNFPConfig(**config.model)
            model = LNFP(**model_config.model_dump())
            return model
        
        case "lnfp-simple":
            model_config = LNFPSimpleConfig(**config.model)
            model = LNFPSimple(**model_config.model_dump())
            return model
        
        case "video-vae":
            model_config = VideoVAEConfig(**config.model)
            model = AutoencoderKL(**model_config.model_dump())
            return model

        case _:
            raise ValueError(f"Model {config.model.name} not supported.")
