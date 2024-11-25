from typing import Any, Literal
from pydantic import BaseModel
from torch import Tensor, nn
import torch

from ._autoencoder import upt_autoencoder_models
from ._approximator import upt_approximator_models
from ...utils import ema_to_model_state_dict



class NextFrameUPTModelConfig(BaseModel):
    upt_autoencoder_type: Literal["UPTAE_M", "UPTAE_T", "UPTAE_S"] = "UPTAE_T"
    upt_autoencoder_kwargs: dict = {}
    upt_autoencoder_checkpoint_path: str
    upt_approximator_type: Literal["UPTA_M", "UPTA_T", "UPTA_S"] = "UPTA_T"
    upt_approximator_kwargs: dict = {}
    device: Literal["cpu", "cuda"] = "cuda"


class NextFrameUPTModel(nn.Module):
    def __init__(
        self,
        upt_autoencoder_type: Literal["UPTAE_M", "UPTAE_T", "UPTAE_S"] = "UPTAE_T",
        upt_autoencoder_kwargs: dict = {},
        upt_autoencoder_checkpoint_path: str | None = None,
        upt_approximator_type: Literal["UPTA_M", "UPTA_T", "UPTA_S"] = "UPTA_T",
        upt_approximator_kwargs: dict = {},
        device: Literal["cpu", "cuda"] = "cuda",
    ):
        super().__init__()
        
        self.autoencoder = upt_autoencoder_models[upt_autoencoder_type](**upt_autoencoder_kwargs)
        self.approximator = upt_approximator_models[upt_approximator_type](**upt_approximator_kwargs)
        
        if upt_autoencoder_checkpoint_path is not None:
            checkpoint = torch.load(upt_autoencoder_checkpoint_path)
            self.vae.load_state_dict(
                ema_to_model_state_dict(checkpoint["ema"]) if "ema" in checkpoint.keys() else checkpoint["model"]
            )
            self.vae.requires_grad_(False)
        
        self.device = device
        
    def forward_train(self, context_frames: Tensor, next_frame: Tensor):
        pass
    
    def rollout_frames(self, x_context: Tensor, n_steps: int,):
        pass