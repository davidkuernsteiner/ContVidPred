from functools import partial

import einops
from kappamodules.layers import Sequential
from kappamodules.transformer import PerceiverPoolingBlock, PrenormBlock, DitPerceiverPoolingBlock, DitBlock
from kappamodules.utils.param_checking import to_2tuple
from kappamodules.vit import VitPatchEmbed, VitPosEmbed2d
from torch import Tensor, nn

from functools import partial

import einops
from kappamodules.layers import ContinuousSincosEmbed, LinearProjection, Sequential
from kappamodules.transformer import PerceiverBlock, DitPerceiverBlock, DitBlock
from kappamodules.vit import VitBlock
from torch import nn
import math


# Taken from https://github.com/BenediktAlkin/upt-minimal/blob/main/upt/models/encoder_image.py
class EncoderImage(nn.Module):
    def __init__(
            self,
            input_dim,
            patch_size,
            resolution,
            enc_dim,
            enc_num_heads,
            enc_depth,
            perc_dim=None,
            perc_num_heads=None,
            num_latent_tokens=None,
            cond_dim=None,
            init_weights="truncnormal",
    ):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        resolution = to_2tuple(resolution)
        self.input_dim = input_dim
        self.patch_size = patch_size
        self.resolution = resolution
        self.enc_dim = enc_dim
        self.enc_depth = enc_depth
        self.enc_num_heads = enc_num_heads
        self.perc_dim = perc_dim
        self.perc_num_heads = perc_num_heads
        self.num_latent_tokens = num_latent_tokens
        self.condition_dim = cond_dim
        self.init_weights = init_weights

        # embed
        self.patch_embed = VitPatchEmbed(
            dim=enc_dim,
            num_channels=input_dim,
            resolution=resolution,
            patch_size=patch_size,
        )
        self.pos_embed = VitPosEmbed2d(seqlens=self.patch_embed.seqlens, dim=enc_dim, is_learnable=False)

        # blocks
        if cond_dim is None:
            block_ctor = PrenormBlock
        else:
            block_ctor = partial(DitBlock, cond_dim=cond_dim)
        self.blocks = Sequential(
            *[
                block_ctor(dim=enc_dim, num_heads=enc_num_heads, init_weights=init_weights)
                for _ in range(enc_depth)
            ],
        )

        # perceiver pooling
        if num_latent_tokens is None:
            self.perceiver = None
        else:
            if cond_dim is None:
                block_ctor = partial(
                    PerceiverPoolingBlock,
                    perceiver_kwargs=dict(
                        kv_dim=enc_dim,
                        init_weights=init_weights,
                    ),
                )
            else:
                block_ctor = partial(
                    DitPerceiverPoolingBlock,
                    perceiver_kwargs=dict(
                        kv_dim=enc_dim,
                        cond_dim=cond_dim,
                        init_weights=init_weights,
                    ),
                )
            self.perceiver = block_ctor(
                dim=perc_dim,
                num_heads=perc_num_heads,
                num_query_tokens=num_latent_tokens,
            )

    def forward(self, input_image, condition=None):
        # check inputs
        assert input_image.ndim == 4, "expected input image of shape (batch_size, num_channels, height, width)"
        if condition is not None:
            assert condition.ndim == 2, "expected shape (batch_size, cond_dim)"

        # pass condition to DiT blocks
        cond_kwargs = {}
        if condition is not None:
            cond_kwargs["cond"] = condition

        # patch_embed
        x = self.patch_embed(input_image)
        # add pos_embed
        x = self.pos_embed(x)
        # flatten
        x = einops.rearrange(x, "b ... d -> b (...) d")

        # transformer
        x = self.blocks(x, **cond_kwargs)

        # perceiver
        if self.perceiver is not None:
            x = self.perceiver(kv=x, **cond_kwargs)

        return x
    

#Taken from https://github.com/BenediktAlkin/upt-minimal/blob/main/upt/models/decoder_perceiver.py
class DecoderPerceiver(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            ndim,
            dim,
            depth,
            num_heads,
            unbatch_mode="dense_to_sparse_unpadded",
            perc_dim=None,
            perc_num_heads=None,
            cond_dim=None,
            init_weights="truncnormal002",
            **kwargs,
    ):
        super().__init__(**kwargs)
        perc_dim = perc_dim or dim
        perc_num_heads = perc_num_heads or num_heads
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.ndim = ndim
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.perc_dim = perc_dim
        self.perc_num_heads = perc_num_heads
        self.cond_dim = cond_dim
        self.init_weights = init_weights
        self.unbatch_mode = unbatch_mode

        # input projection
        self.input_proj = LinearProjection(input_dim, dim, init_weights=init_weights, optional=True)

        # blocks
        if cond_dim is None:
            block_ctor = VitBlock
        else:
            block_ctor = partial(DitBlock, cond_dim=cond_dim)
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

        # prepare perceiver
        self.pos_embed = ContinuousSincosEmbed(
            dim=perc_dim,
            ndim=ndim,
        )
        if cond_dim is None:
            block_ctor = PerceiverBlock
        else:
            block_ctor = partial(DitPerceiverBlock, cond_dim=cond_dim)

        # decoder
        self.query_proj = nn.Sequential(
            LinearProjection(perc_dim, perc_dim, init_weights=init_weights),
            nn.GELU(),
            LinearProjection(perc_dim, perc_dim, init_weights=init_weights),
        )
        self.perc = block_ctor(dim=perc_dim, kv_dim=dim, num_heads=perc_num_heads, init_weights=init_weights)
        self.pred = nn.Sequential(
            nn.LayerNorm(perc_dim, eps=1e-6),
            LinearProjection(perc_dim, output_dim, init_weights=init_weights),
        )

    def forward(self, x, output_pos, condition=None):
        # check inputs
        assert x.ndim == 3, "expected shape (batch_size, num_latent_tokens, dim)"
        assert output_pos.ndim == 3, "expected shape (batch_size, num_outputs, dim) num_outputs might be padded"
        if condition is not None:
            assert condition.ndim == 2, "expected shape (batch_size, cond_dim)"

        # pass condition to DiT blocks
        cond_kwargs = {}
        if condition is not None:
            cond_kwargs["cond"] = condition

        # input projection
        x = self.input_proj(x)

        # apply blocks
        x = self.blocks(x, **cond_kwargs)

        # create query
        query = self.pos_embed(output_pos)
        query = self.query_proj(query)

        x = self.perc(q=query, kv=x, **cond_kwargs)
        x = self.pred(x)
        if self.unbatch_mode == "dense_to_sparse_unpadded":
            # dense to sparse where no padding needs to be considered
            x = einops.rearrange(
                x,
                "batch_size seqlen dim -> (batch_size seqlen) dim",
            )
        elif self.unbatch_mode == "image":
            # rearrange to square image
            height = math.sqrt(x.size(1))
            assert height.is_integer()
            x = einops.rearrange(
                x,
                "batch_size (height width) dim -> batch_size dim height width",
                height=int(height),
            )
        else:
            raise NotImplementedError(f"invalid unbatch_mode '{self.unbatch_mode}'")

        return x
    

# Adapted from https://github.com/BenediktAlkin/upt-minimal/blob/main/upt/models/upt_image_autoencoder.py
class UPTImageAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 3,
        patch_size: int = 2,
        resolution: int = 32,
        latent_dim: int = 96,
        num_heads: int = 4,
        depth: int = 4,
        perc_dim: int = 96,
        perc_num_heads: int = 4,
        num_latent_tokens: int = 64,
    ):
        super().__init__()
        self.encoder = EncoderImage(
            input_dim=input_dim,
            patch_size=patch_size,
            resolution=resolution,
            enc_dim=latent_dim,
            enc_num_heads=num_heads,
            enc_depth=depth,
            perc_dim=perc_dim,
            perc_num_heads=perc_num_heads,
            num_latent_tokens=num_latent_tokens,
        )
        self.decoder = DecoderPerceiver(
            input_dim=latent_dim,
            output_dim=input_dim,
            ndim=2,
            dim=latent_dim,
            depth=depth,
            num_heads=num_heads,
            perc_dim=perc_dim,
            perc_num_heads=perc_num_heads,
            unbatch_mode="image",
        )

    def forward(self, x: Tensor, output_pos: Tensor):
        latent = self.encoder(x)
        pred = self.decoder(latent, output_pos=output_pos)

        return pred
    
    def encode(self, x: Tensor):
        return self.encoder(x)
    
    def decode(self, latent: Tensor, output_pos: Tensor):
        return self.decoder(latent, output_pos=output_pos)
    

def UPTAE_M(**kwargs):
    return UPTImageAutoencoder(
        input_dim=3, 
        patch_size=2, 
        latent_dim=96, 
        num_heads=2, 
        depth=2, 
        perc_dim=96, 
        perc_num_heads=2,
        num_latent_tokens=8,
        **kwargs
        )
    
def UPTAE_T(**kwargs):
    return UPTImageAutoencoder(
        input_dim=3, 
        patch_size=2, 
        latent_dim=96, 
        num_heads=4, 
        depth=4, 
        perc_dim=96, 
        perc_num_heads=4,
        num_latent_tokens=16,
        **kwargs,
        )
    
def UPTAE_S(**kwargs):
    return UPTImageAutoencoder(
        input_dim=3, 
        patch_size=2, 
        latent_dim=192, 
        num_heads=4, 
        depth=4, 
        perc_dim=192, 
        perc_num_heads=4,
        num_latent_tokens=32,
        **kwargs,
        )
    
upt_autoencoder_models = {
    "UPTAE_M": UPTAE_M,
    "UPTAE_T": UPTAE_T,
    "UPTAE_S": UPTAE_S
}