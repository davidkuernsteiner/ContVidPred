# Modified from OpenAI's diffusion repos
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py

from typing import Literal
import torch

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps
from .diffusion_utils import enforce_zero_terminal_snr


def create_diffusion(
    timestep_respacing: str,
    noise_schedule: str = "linear", 
    use_kl: bool = False,
    predict_target: Literal["epsilon", "velocity", "xstart", "xprev"] = "velocity",
    sigma_type: Literal["fixed_small", "fixed_large", "learned"] = "fixed_small",
    rescale_learned_sigmas: bool = False,
    rescale_betas_zero_snr: bool = True,
    diffusion_steps: int = 1000
):
    if rescale_betas_zero_snr:
        betas = enforce_zero_terminal_snr(torch.Tensor(gd.get_named_beta_schedule(noise_schedule, diffusion_steps)))
    else:
        betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)
        
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
        
    if timestep_respacing is None or timestep_respacing == "":
        timestep_respacing = [diffusion_steps]
        
    match predict_target:
        case "epsilon":
            model_mean_type = gd.ModelMeanType.EPSILON
        case "velocity":
            model_mean_type = gd.ModelMeanType.VELOCITY
        case "xstart":
            model_mean_type = gd.ModelMeanType.START_X
        case "xprev":
            model_mean_type = gd.ModelMeanType.PREVIOUS_X
    
    match sigma_type:
        case "fixed_small":
            model_var_type = gd.ModelVarType.FIXED_SMALL
        case "fixed_large":
            model_var_type = gd.ModelVarType.FIXED_LARGE
        case "learned":
            model_var_type = gd.ModelVarType.LEARNED_RANGE
            
    return SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        model_mean_type=model_mean_type,
        model_var_type=model_var_type,
        loss_type=loss_type,
    )