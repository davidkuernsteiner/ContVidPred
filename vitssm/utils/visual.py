from typing import Sequence, Union
from torch import Tensor
from einops import rearrange
import torch
from PIL import Image
import matplotlib.pyplot as plt



def unnormalize(tensor: torch.Tensor):
    return tensor * 0.5 + 0.5


def model_output_to_image(x: torch.Tensor):
    return Image.fromarray(rearrange(unnormalize(x.detach().cpu()).numpy() * 255, "c h w -> h w c").astype("uint8"))


def model_output_to_video(x: torch.Tensor):
    return (unnormalize(x.detach().cpu()).numpy() * 255).astype("uint8")


def display_video_frames_in_grid(video: torch.Tensor):
    frames = []
    frame_indices = []
    
    for i, frame in enumerate(video):
        # Read each frame one by one
        frame = video[i]
        # Convert BGR (OpenCV) to RGB (Matplotlib)
        frames.append(model_output_to_image(frame))
        frame_indices.append(i)  # Save the frame index

    # Create a grid of frames with their index labels
    fig, axes = plt.subplots(int(len(video)**0.5), int(len(video)**0.5), figsize=(12, 8))
    for ax, frame, idx in zip(axes.ravel(), frames, frame_indices):
        ax.imshow(frame)
        ax.set_title(f"Frame {idx}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()