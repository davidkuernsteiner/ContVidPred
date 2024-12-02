from omegaconf import DictConfig
from torch.nn import Module

from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

from .vae import VideoVAEConfig
#from .latte import LatteDiffusionModelConfig, LatteDiffusionModel
from .dit import NextFrameDiTModelConfig, NextFrameDiTModel
from .unet import UncondUNetModel, UncondUNetModelConfig, NextFrameUNetModel, NextFrameUNetModelConfig, BasicNextFrameUNetModel, BasicNextFrameUNetModelConfig
from .upt import NextFrameUPTModel, NextFrameUPTModelConfig, UPTImageAutoencoder, UPTImageAutoencoderConfig, UPTVideoAutoencoder, UPTVideoAutoencoderConfig


def build_model(config: DictConfig) -> Module:
    """Builds model given config."""
    match config.name:
        case "vae":
            model_config = VideoVAEConfig(**config)
            model = AutoencoderKL(**model_config.model_dump())
            return model
        
        #case "latte":
        #    model_config = LatteDiffusionModelConfig(**config)
        #    model = LatteDiffusionModel(**model_config.model_dump())
        #    return model
        
        case "dit":
            model_config = NextFrameDiTModelConfig(**config)
            model = NextFrameDiTModel(**model_config.model_dump())
            return model
        
        case "unet":
            model_config = UncondUNetModelConfig(**config)
            model = UncondUNetModel(**model_config.model_dump())
            return model
        
        case "unet_next_frame":
            model_config = NextFrameUNetModelConfig(**config)
            model = NextFrameUNetModel(**model_config.model_dump())
            return model
        
        case "basic_unet_next_frame":
            model_config = BasicNextFrameUNetModelConfig(**config)
            model = BasicNextFrameUNetModel(**model_config.model_dump())
            return model
        
        case "upt_ae":
            model_config = UPTImageAutoencoderConfig(**config)
            model = UPTImageAutoencoder(**model_config.model_dump())
            return model
        
        case "upt_vae":
            model_config = UPTVideoAutoencoderConfig(**config)
            model = UPTVideoAutoencoder(**model_config.model_dump())
            return model
        
        case "upt_next_frame":
            model_config = NextFrameUPTModelConfig(**config)
            model = NextFrameUPTModel(**model_config.model_dump())
            return model
        
        case _:
            raise ValueError(f"Model {config.name} not supported.")
