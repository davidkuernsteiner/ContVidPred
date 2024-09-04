import argparse
import os
from pathlib import Path
from tqdm import tqdm

from PIL import ImageColor
from torchvision.io import write_video

from vitssm.data.mdsprites.generators import generate_videos
from vitssm.data.mdsprites.shapes import Circle, Rectangle, Triangle


parser = argparse.ArgumentParser(description='Generate VMdsprites')

parser.add_argument("-o", "--output", help="Output directory", required=True, type=str)
parser.add_argument("-m", "--masks", help="Whether to generate masks", action="store_true")
parser.add_argument("--resolution", help="Resolution of the video", default=64, type=int)
parser.add_argument("--n_shapes", help="Maximum number of shapes", default=3, type=int)
parser.add_argument("--n_videos", help="Number of videos to generate", default=100, type=int)
parser.add_argument("--video_length", help="Length of the video", default=100, type=int)
parser.add_argument("--background", help="Background color", default="black", type=str)

args = parser.parse_args()

data_folder = Path(args.output, "VMDsprites")
os.makedirs(data_folder / "videos", exist_ok=True)
os.makedirs(data_folder / "masks", exist_ok=True)

object_colors = ImageColor.colormap
if args.background in object_colors:
    object_colors.pop(args.background)

video_generator = iter(generate_videos(
    resolution=args.resolution,
    masks=args.masks,
    shapes=[Circle, Rectangle, Triangle],
    n_shapes=args.n_shapes,
    colors=list(object_colors.keys()),
    background=args.background,
    video_length=args.video_length,
))

for i in tqdm(
    range(args.n_videos),
    desc="Generating videos",
    total=args.n_videos,
):
    video, masks = next(video_generator)
    
    write_video(data_folder / "videos" / f"video_{i}.avi", video, fps=10)
    if args.masks:
        for j, mask in enumerate(masks):
            os.makedirs(data_folder / "masks" / f"{i}", exist_ok=True)
            write_video(data_folder / "masks" / f"video_{i}_mask_{j}.avi", mask, fps=10)
