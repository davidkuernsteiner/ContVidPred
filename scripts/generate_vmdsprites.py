import argparse
import os
import random
import shutil
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import pandas as pd

from PIL import ImageColor
from torchvision.io import write_video

from vitssm.data.mdsprites.generators import generate_videos
from vitssm.data.mdsprites.shapes import Circle, Rectangle, Triangle

def split_list(lst: list, percentage: float) -> tuple[list, list]:
    # Shuffle the list to randomize order
    random.shuffle(lst)
    
    # Calculate split index
    split_index = int(len(lst) * percentage)

    # Split the list
    part1 = lst[:split_index]
    part2 = lst[split_index:]

    return part1, part2

def main(
    output_dir: str,
    generate_masks: bool,
    resolution: int,
    n_shapes: int,
    n_videos: int,
    video_length: int,
    background: str,
    exist_ok: bool,
    n_folds: int = 5,
    train_ratio: float = 0.8,
) -> None:
    
    if exist_ok:
        data_folder = Path(output_dir) / f"VMDsprites_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    else:
        data_folder = Path(output_dir) / "VMDsprites"
        if data_folder.exists():
            shutil.rmtree(data_folder)
        
    os.makedirs(data_folder / "videos", exist_ok=True)
    os.makedirs(data_folder / "masks", exist_ok=True)

    object_colors = ImageColor.colormap
    if background in object_colors:
        object_colors.pop(background)

    video_generator = iter(generate_videos(
        resolution=resolution,
        generate_masks=generate_masks,
        shapes=[Circle, Rectangle, Triangle],
        n_shapes=(1, n_shapes),
        colors=list(object_colors.keys()),
        background=background,
        video_length=video_length,
    ))

    video_paths = []
    for i in tqdm(
        range(n_videos),
        desc="Generating videos",
        total=n_videos,
    ):
        video, masks = next(video_generator)
        video_path = data_folder / "videos" / f"video_{i}.avi"
        
        write_video(video_path, video, fps=1)
        video_paths.append(video_path)
        
        if generate_masks:
            for j, mask in enumerate(masks):
                mask_dir = data_folder / "masks" / f"{i}"
                os.makedirs(mask_dir, exist_ok=True)
                write_video(mask_dir / f"video_{i}_mask_{j}.avi", mask, fps=1)
    
    fold_dir = data_folder / "folds"
    os.makedirs(fold_dir, exist_ok=True)
    for fold_idx in range(n_folds):
        train_split, test_split = split_list(video_paths, train_ratio)       
        pd.DataFrame({"path": train_split}).to_csv(fold_dir / f"train_{fold_idx}.csv", index=False)
        pd.DataFrame({"path": test_split}).to_csv(fold_dir / f"test_{fold_idx}.csv", index=False)   


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate VMdsprites')

    parser.add_argument("-o", "--output", help="Output directory", required=True, type=str)
    parser.add_argument("-m", "--masks", help="Whether to generate masks", action="store_true")
    parser.add_argument("--resolution", help="Resolution of the video", default=64, type=int)
    parser.add_argument("--n_shapes", help="Maximum number of shapes", default=4, type=int)
    parser.add_argument("--n_videos", help="Number of videos to generate", default=100, type=int)
    parser.add_argument("--video_length", help="Length of the video", default=100, type=int)
    parser.add_argument("--background", help="Background color", default="black", type=str)
    parser.add_argument("--exist_ok", help="Allow multiple VMDsprites datasets", action="store_true")
    parser.add_argument("--n_folds", help="Number of folds", default=5, type=int)
    parser.add_argument("--train_ratio", help="Train ratio", default=0.8, type=float)

    args = parser.parse_args()

    main(
        output_dir=args.output,
        generate_masks=args.masks,
        resolution=args.resolution,
        n_shapes=args.n_shapes,
        n_videos=args.n_videos,
        video_length=args.video_length,
        background=args.background,
        exist_ok=args.exist_ok,
        n_folds=args.n_folds,
        train_ratio=args.train_ratio,
    )
