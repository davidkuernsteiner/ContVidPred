from omegaconf import DictConfig
from torch.nn import Module

from .next_frame_prediction import LatentNextFramePrediction, LatentNextFramePredictionConfig


def build_model(config: DictConfig) -> Module:
    """Builds model given config."""
    match config.model.name:
        
        case "latent_nextframe_attention":
            model_config = LatentNextFramePredictionConfig(**config.model)
            model = LatentNextFramePrediction(**model_config.model_dump())
            return model
        
        case _:
            raise ValueError(f"Model {config.model.name} not supported.")