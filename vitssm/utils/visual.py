from typing import Sequence, Union
from torch import Tensor
from einops import rearrange
import torch
import numpy as np
import cv2
import imageio.v3 as imageio
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


def save_videos_as_grid_gif(video_list, grid_shape, output_path, fps=10):
    """
    Save multiple videos in a grid layout as a GIF.

    Args:
        video_list (list of np.ndarray): List of videos, each with shape (num_frames, height, width, channels).
        grid_shape (tuple): Shape of the grid (rows, cols).
        output_path (str): Path to save the GIF.
        fps (int): Frames per second for the GIF.
    """
    # Ensure the number of videos matches the grid
    assert len(video_list) == grid_shape[0] * grid_shape[1], \
        "Number of videos must match grid dimensions (rows x cols)."
    
    # Determine the maximum number of frames across all videos
    max_frames = max(video.shape[0] for video in video_list)
    
    # Pad videos to the same number of frames if necessary
    padded_videos = [
        np.pad(video, ((0, max_frames - video.shape[0]), (0, 0), (0, 0), (0, 0)), mode='constant') 
        for video in video_list
    ]
    
    # Extract height and width of individual videos
    video_height, video_width = padded_videos[0].shape[1:3]
    
    # Create a blank frame for the grid
    grid_height = grid_shape[0] * video_height
    grid_width = grid_shape[1] * video_width
    grid_frames = []

    # Create grid frames by combining individual frames
    for frame_idx in range(max_frames):
        grid_frame = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
        for i, video in enumerate(padded_videos):
            row = i // grid_shape[1]
            col = i % grid_shape[1]
            start_y = row * video_height
            start_x = col * video_width
            grid_frame[start_y:start_y+video_height, start_x:start_x+video_width] = video[frame_idx]
        grid_frames.append(grid_frame)
    
    # Save the grid frames as a GIF
    imageio.imwrite(output_path, grid_frames, fps=fps)
    print(f"Saved GIF to {output_path}")