import argparse
import os
from pathlib import Path
from tqdm import tqdm

from PIL import ImageColor
from torchvision.io import write_video

from vitssm.data.mdsprites.generators import generate_videos
from vitssm.data.mdsprites.shapes import Circle, Rectangle, Triangle



def main(
    output_dir: str,
    generate_masks: bool,
    resolution: int,
    n_shapes: int,
    n_videos: int,
    video_length: int,
    background: str
) -> None:
    data_folder = Path(output_dir, "VMDsprites")
    os.makedirs(data_folder / "videos", exist_ok=True)
    os.makedirs(data_folder / "masks", exist_ok=True)

    object_colors = ImageColor.colormap
    if background in object_colors:
        object_colors.pop(background)

    video_generator = iter(generate_videos(
        resolution=resolution,
        generate_masks=generate_masks,
        shapes=[Circle, Rectangle, Triangle],
        n_shapes=n_shapes,
        colors=list(object_colors.keys()),
        background=background,
        video_length=video_length,
    ))

    for i in tqdm(
        range(n_videos),
        desc="Generating videos",
        total=n_videos,
    ):
        video, masks = next(video_generator)

        write_video(data_folder / "videos" / f"video_{i}.avi", video, fps=10)
        if generate_masks:
            for j, mask in enumerate(masks):
                os.makedirs(data_folder / "masks" / f"{i}", exist_ok=True)
                write_video(data_folder / "masks" / f"video_{i}_mask_{j}.avi", mask, fps=10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate VMdsprites')

    parser.add_argument("-o", "--output", help="Output directory", required=True, type=str)
    parser.add_argument("-m", "--masks", help="Whether to generate masks", action="store_true")
    parser.add_argument("--resolution", help="Resolution of the video", default=64, type=int)
    parser.add_argument("--n_shapes", help="Maximum number of shapes", default=3, type=int)
    parser.add_argument("--n_videos", help="Number of videos to generate", default=100, type=int)
    parser.add_argument("--video_length", help="Length of the video", default=100, type=int)
    parser.add_argument("--background", help="Background color", default="black", type=str)

    args = parser.parse_args()

    main(
        output_dir=args.output,
        generate_masks=args.masks,
        resolution=args.resolution,
        n_shapes=args.n_shapes,
        n_videos=args.n_videos,
        video_length=args.video_length,
        background=args.background
    )
