from functools import partial

from einops import rearrange
import torch
from kappamodules.layers import LinearProjection, Sequential
from kappamodules.transformer import DitBlock, PrenormBlock, PerceiverPoolingBlock, DitPerceiverPoolingBlock
from kappamodules.vit import VitPosEmbed1d
from torch import nn


#Adapted from https://github.com/BenediktAlkin/upt-minimal/blob/main/upt/models/approximator.py
class UPTContextPercApproximator(nn.Module):
    def __init__(
            self,
            input_dim=96,
            context_length=4,
            num_output_tokens=32,
            depth=4,
            num_heads=4,
            dim=None,
            cond_dim=None,
            init_weights="truncnormal002",
            **kwargs,
    ):
        super().__init__(**kwargs)
        dim = dim or input_dim
        self.num_output_tokens = num_output_tokens
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.cond_dim = cond_dim
        self.init_weights = init_weights

        # project
        self.input_proj = LinearProjection(input_dim, dim, init_weights=init_weights, optional=True)

        # blocks
        if cond_dim is None:
            block_ctor = PrenormBlock
            block_pool = PerceiverPoolingBlock
        else:
            block_ctor = partial(DitBlock, cond_dim=cond_dim)
            block_pool = partial(DitPerceiverPoolingBlock, perceiver_kwargs={"cond_dim": cond_dim})
            
        # Temporal pooling
        self.temp_pos_embed = VitPosEmbed1d(seqlens=(context_length, ), dim=dim, is_learnable=False)
        self.temp_pool = block_pool(
            dim=dim, 
            num_heads=num_heads,
            num_query_tokens=num_output_tokens,
        )
            
        self.blocks = Sequential(
            *[
                block_ctor(
                    dim=dim,
                    num_heads=num_heads,
                    init_weights=init_weights,
                )
                for _ in range(depth)
            ],
        )

    def forward(self, x, condition=None):
        # check inputs
        assert x.ndim == 4, "expected shape (batch_size, context_length, num_latent_tokens, dim)"
        if condition is not None:
            assert condition.ndim == 2, "expected shape (batch_size, cond_dim)"
            
        # pass condition to DiT blocks
        cond_kwargs = {}
        if condition is not None:
            cond_kwargs["cond"] = condition

        # project to decoder dim
        x = self.input_proj(x)
        
        b, c, t, d = x.shape
        # temporal embedding + pooling
        x = rearrange(x, "bs cl nt di -> (bs nt) cl di")
        x = self.temp_pos_embed(x)
        x = rearrange(x, "(bs nt) cl di -> bs (cl nt) di", bs=b, cl=c, nt=t)
        x = self.temp_pool(x)

        # apply blocks
        x = self.blocks(x, **cond_kwargs)

        return x
    
    
#Adapted from https://github.com/BenediktAlkin/upt-minimal/blob/main/upt/models/approximator.py
class UPTContextCatApproximator(nn.Module):
    def __init__(
            self,
            input_dim=96,
            context_length=4,
            depth=4,
            num_heads=4,
            dim=None,
            cond_dim=None,
            init_weights="truncnormal002",
            **kwargs,
    ):
        super().__init__(**kwargs)
        dim = dim or input_dim
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.cond_dim = cond_dim
        self.init_weights = init_weights

        # project
        self.input_proj = LinearProjection(input_dim * context_length, dim, init_weights=init_weights)

        # blocks
        if cond_dim is None:
            block_ctor = PrenormBlock
        else:
            block_ctor = partial(DitBlock, cond_dim=cond_dim)
            
        # Temporal pooling
        #self.final_layer = Sequential(
        #    nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6),
        #    LinearProjection(context_length * dim, dim, bias=True, init_weights=init_weights),
        #)
            
        self.blocks = Sequential(
            *[
                block_ctor(
                    dim=dim,
                    num_heads=num_heads,
                    init_weights=init_weights,
                )
                for _ in range(depth)
            ],
        )

    def forward(self, x, condition=None):
        # check inputs
        assert x.ndim == 4, "expected shape (batch_size, context_length, num_latent_tokens, dim)"
        if condition is not None:
            assert condition.ndim == 2, "expected shape (batch_size, cond_dim)"
            
        # pass condition to DiT blocks
        cond_kwargs = {}
        if condition is not None:
            cond_kwargs["cond"] = condition

        # project to decoder dim
        x = rearrange(x, "bs cl nt di -> bs nt (cl di)")
        x = self.input_proj(x)
        
        # concat context tokens along latent dimension
        # apply blocks
        x = self.blocks(x, **cond_kwargs)
        # project down to single latent representation
        #x = self.final_layer(x)

        return x
    

def UPTCoPeA_M(**kwargs):
    return UPTContextPercApproximator(
        input_dim=96,
        depth=4,
        num_heads=2,
        dim=96,
        num_output_tokens=8,
        **kwargs,
    )
    
def UPTCoPeA_T(**kwargs):
    return UPTContextPercApproximator(
        input_dim=96,
        depth=8,
        num_heads=4,
        dim=96,
        num_output_tokens=16,
        **kwargs,
    )
    
def UPTCoPeA_S(**kwargs):
    return UPTContextPercApproximator(
        input_dim=192,
        depth=12,
        num_heads=6,
        dim=192,
        num_output_tokens=32,
        **kwargs,
    )
    
def UPTCoCaA_M(**kwargs):
    return UPTContextCatApproximator(
        input_dim=96,
        depth=4,
        num_heads=2,
        dim=96,
        **kwargs,
    )
    
def UPTCoCaA_T(**kwargs):
    return UPTContextCatApproximator(
        input_dim=96,
        depth=8,
        num_heads=4,
        dim=96,
        **kwargs,
    )
    
def UPTCoCaA_S(**kwargs):
    return UPTContextCatApproximator(
        input_dim=192,
        depth=12,
        num_heads=6,
        dim=192,
        **kwargs,
    )
    

upt_approximator_models = {
    "UPTCoPeA_M": UPTCoPeA_M,
    "UPTCoPeA_T": UPTCoPeA_T,
    "UPTCoPeA_S": UPTCoPeA_S,
    "UPTCoCaA_M": UPTCoCaA_M,
    "UPTCoCaA_T": UPTCoCaA_T,
    "UPTCoCaA_S": UPTCoCaA_S,
}