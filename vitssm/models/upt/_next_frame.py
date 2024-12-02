from typing import Any, Literal
from einops import rearrange, repeat
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
    upt_approximator_type: Literal["UPTCoPeA_M", "UPTCoPeA_T", "UPTCoPeA_S", "UPTCoCaA_M", "UPTCoCaA_T", "UPTCoCaA_S"] = "UPTCoCaA_M"
    upt_approximator_kwargs: dict = {}
    device: Literal["cpu", "cuda"] = "cuda"


class NextFrameUPTModel(nn.Module):
    def __init__(
        self,
        upt_autoencoder_type: Literal["UPTAE_M", "UPTAE_T", "UPTAE_S"] = "UPTAE_T",
        upt_autoencoder_kwargs: dict = {},
        upt_autoencoder_checkpoint_path: str | None = None,
        upt_approximator_type: Literal["UPTCoPeA_M", "UPTCoPeA_T", "UPTCoPeA_S", "UPTCoCaA_M", "UPTCoCaA_T", "UPTCoCaA_S"] = "UPTCoCaA_M",
        upt_approximator_kwargs: dict = {},
        device: Literal["cpu", "cuda"] = "cuda",
    ):
        super().__init__()
        
        self.autoencoder = upt_autoencoder_models[upt_autoencoder_type](**upt_autoencoder_kwargs)
        self.approximator = upt_approximator_models[upt_approximator_type](**upt_approximator_kwargs)
        
        if upt_autoencoder_checkpoint_path is not None:
            checkpoint = torch.load(upt_autoencoder_checkpoint_path)
            self.autoencoder.load_state_dict(
                ema_to_model_state_dict(checkpoint["ema"]) if "ema" in checkpoint.keys() else checkpoint["model"]
            )
            self.autoencoder.requires_grad_(False)
        
        self.device = device
        
    def forward_train(self, context_frames: Tensor, next_frame: Tensor):
        x = torch.cat((context_frames, next_frame), dim=1)
        bs, cl, _, _, _ = x.shape
        
        # Encode frames
        x = rearrange(x, 'bs cl ch ht wt -> (bs cl) ch ht wt')
        with torch.no_grad():
            x = self.autoencoder.encode(x)  
        x = rearrange(x, '(bs cl) nt di -> bs cl nt di', bs=bs, cl=cl)
 
        x_context, x_next = torch.split(x, [cl - 1,  1], dim=1)
        
        x_next_pred = self.approximator(x_context)
        
        return x_next_pred, x_next.squeeze(1)
    
    def rollout_frames(self, x_context: Tensor, n_steps: int,):
        bs, cl, ch, ht, wt = x_context.shape
        
        # Encode context frames
        x_context = rearrange(x_context, 'bs cl ch ht wt -> (bs cl) ch ht wt')
        x_context = self.autoencoder.encode(x_context)  
        x_context = rearrange(x_context, '(bs cl) nt di -> bs cl nt di', bs=bs, cl=cl)
        
        x = []
        for _ in range(n_steps):
            x_next_pred = self.approximator(x_context).unsqueeze(1)
            x.append(x_next_pred.clone())
            x_context = torch.cat((x_context[:, 1:], x_next_pred), dim=1)
        
        x = torch.cat(x, dim=1)
        bs, nf, nt, di = x.shape
        
        x = rearrange(x, "bs nf nt di -> (bs nf) nt di")
        
        output_pos = rearrange(
            torch.stack(torch.meshgrid([torch.arange(ht), torch.arange(ht)], indexing="ij")),
            "ndim height width -> (height width) ndim",
        ).float().to(self.device)
        output_pos = output_pos / (ht - 1) * 1000
        
        x = self.autoencoder.decode(x, output_pos=repeat(output_pos, "... -> b ...", b=bs*nf))
        x = rearrange(x, "(bs nf) ch ht wt -> bs nf ch ht wt", bs=bs, nf=nf)
        
        return x
        