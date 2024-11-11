from typing import Any, Optional, Union
from pydantic import BaseModel

from diffusers import UNet2DModel
from diffusers.models.unets.unet_2d import UNet2DOutput
import torch



class UNet2DNextFrameModel(UNet2DModel):
    def __init__(**kwargs):
        super().__init__(**kwargs)
    
    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        context: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        if context is not None:
            sample = torch.cat((context, sample), dim=1)
        
        return super().forward(
            sample=sample,
            timestep=timestep,
            class_labels=class_labels,
            return_dict=True,
        ).sample
        
    def forward_with_cfg(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        context: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.Tensor] = None,
        cfg_scale: float = 7.5,
        cfg_rescale_factor: float = 0.7,
    ) -> torch.Tensor:
        
        half = sample[: len(sample) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, timestep=timestep, context=context, class_labels=class_labels)
        
        x_pos, x_neg = torch.split(model_out, len(model_out) // 2, dim=0)
        x_cfg = x_neg + cfg_scale * (x_pos - x_neg)
        
        # Apply improved cfg https://arxiv.org/pdf/2305.08891
        std_pos = x_pos.std([1,2,3], keepdim=True)
        std_cfg = x_cfg.std([1,2,3], keepdim=True)
        factor = std_pos / std_cfg
        factor = cfg_rescale_factor * factor + (1- cfg_rescale_factor)
        x_cfg = x_cfg * factor
        x_cfg = torch.cat([x_cfg, x_cfg], dim=0)
        
        return x_cfg


def UNet_B(**kwargs):
    return UNet2DNextFrameModel(
        down_block_types=("AttnDownBlock2D", "AttnDownBlock2D", "DownBlock2D",),
        mid_block_type="UNetMidBlock2D",
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D"),
        block_out_channels=(128, 256, 512),
        layers_per_block=2,
        cross_attention_dim=512,
        **kwargs,
    )

def UNet_S(**kwargs):
    return UNet2DNextFrameModel(
        down_block_types=("AttnDownBlock2D", "AttnDownBlock2D", "DownBlock2D",),
        mid_block_type="UNetMidBlock2D",
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D"),
        block_out_channels=(64, 128, 256),
        layers_per_block=2,
        cross_attention_dim=256,
        **kwargs,
    )

def UNet_T(**kwargs):
    return UNet2DNextFrameModel(
        down_block_types=("AttnDownBlock2D", "DownBlock2D",),
        mid_block_type="UNetMidBlock2D",
        up_block_types=("UpBlock2D", "AttnUpBlock2D"),
        block_out_channels=(128, 256),
        layers_per_block=2,
        cross_attention_dim=256,
        **kwargs,
    )

def UNet_M(**kwargs):
    return UNet2DNextFrameModel(
        down_block_types=("AttnDownBlock2D", "DownBlock2D",),
        mid_block_type="UNetMidBlock2D",
        up_block_types=("UpBlock2D", "AttnUpBlock2D"),
        block_out_channels=(64, 128),
        layers_per_block=1,
        cross_attention_dim=128,
        **kwargs,
    )
    
UNet_models = {
    "UNet_B": UNet_B,
    "UNet_S": UNet_S,
    "UNet_T": UNet_T,
    "UNet_M": UNet_M,
}