from typing import Any, Optional, Union
from pydantic import BaseModel

from diffusers import UNet2DConditionModel
import torch


#class UNet2DConditionConfig(BaseModel):
#    sample_size: Optional[int] = None
#    in_channels: int = 4
#    out_channels: int = 4
#    center_input_sample: bool = False
#    flip_sin_to_cos: bool = True
#    freq_shift: int = 0
#    down_block_types: Tuple[str] = (
#        "CrossAttnDownBlock2D",
#        "CrossAttnDownBlock2D",
#        "CrossAttnDownBlock2D",
#        "DownBlock2D",
#    ),
#    mid_block_type: Optional[str] = "UNetMidBlock2DCrossAttn",
#    up_block_types: Tuple[str] = ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
#    only_cross_attention: Union[bool, Tuple[bool]] = False,
#    block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
#    layers_per_block: Union[int, Tuple[int]] = 2,
#    downsample_padding: int = 1,
#    mid_block_scale_factor: float = 1,
#    dropout: float = 0.0,
#    act_fn: str = "silu",
#    norm_num_groups: Optional[int] = 32,
#    norm_eps: float = 1e-5,
#    cross_attention_dim: Union[int, Tuple[int]] = 1280,
#    transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple]] = 1,
#    reverse_transformer_layers_per_block: Optional[Tuple[Tuple[int]]] = None,
#    encoder_hid_dim: Optional[int] = None,
#    encoder_hid_dim_type: Optional[str] = None,
#    attention_head_dim: Union[int, Tuple[int]] = 8,
#    num_attention_heads: Optional[Union[int, Tuple[int]]] = None,
#    dual_cross_attention: bool = False,
#    use_linear_projection: bool = False,
#    class_embed_type: Optional[str] = None,
#    addition_embed_type: Optional[str] = None,
#    addition_time_embed_dim: Optional[int] = None,
#    num_class_embeds: Optional[int] = None,
#    upcast_attention: bool = False,
#    resnet_time_scale_shift: str = "default",
#    resnet_skip_time_act: bool = False,
#    resnet_out_scale_factor: float = 1.0,
#    time_embedding_type: str = "positional",
#    time_embedding_dim: Optional[int] = None,
#    time_embedding_act_fn: Optional[str] = None,
#    timestep_post_act: Optional[str] = None,
#    time_cond_proj_dim: Optional[int] = None,
#    conv_in_kernel: int = 3,
#    conv_out_kernel: int = 3,
#    projection_class_embeddings_input_dim: Optional[int] = None,
#    attention_type: str = "default",
#    class_embeddings_concat: bool = False,
#    mid_block_only_cross_attention: Optional[bool] = None,
#    cross_attention_norm: Optional[str] = None,
#    addition_embed_type_num_heads: int = 64,


class UNet2DNextFrameModel(UNet2DConditionModel):
    def __init__(**kwargs):
        super().__init__(**kwargs)
    
    def forward(
        self,
        sample: torch.Tensor,
        context: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[dict[str, Any]] = None,
        added_cond_kwargs: Optional[dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        down_intrablock_additional_residuals: Optional[tuple[torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet2DConditionOutput, tuple]:
        
        sample = torch.cat((context, sample), dim=1)
        
        return super().forward(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            class_labels=class_labels,
            timestep_cond=timestep_cond,
            attention_mask=attention_mask,
            cross_attention_kwargs=cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual,
            down_intrablock_additional_residuals=down_intrablock_additional_residuals,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=return_dict,
        )


def UNet_B(**kwargs):
    return UNet2DConditionModel(
        down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D",),
        mid_block_type="UNetMidBlock2DCrossAttn",
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        block_out_channels=(128, 256, 512),
        layers_per_block=2,
        cross_attention_dim=512,
        **kwargs,
    )

def UNet_S(**kwargs):
    return UNet2DConditionModel(
        down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D",),
        mid_block_type="UNetMidBlock2DCrossAttn",
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        block_out_channels=(64, 128, 256),
        layers_per_block=2,
        cross_attention_dim=256,
        **kwargs,
    )

def UNet_T(**kwargs):
    return UNet2DConditionModel(
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D",),
        mid_block_type="UNetMidBlock2DCrossAttn",
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        block_out_channels=(128, 256),
        layers_per_block=2,
        cross_attention_dim=256,
        **kwargs,
    )

def UNet_M(**kwargs):
    return UNet2DConditionModel(
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D",),
        mid_block_type="UNetMidBlock2DCrossAttn",
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
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