import argparse

from PIL import ImageColor

from vitssm.data.mdsprites.generators import generate_videos
from vitssm.data.mdsprites.shapes import Circle, Rectangle, Triangle


parser = argparse.ArgumentParser(description='Generate VMdsprites')

parser.add_argument("-o", "--output", help="Output file", required=True, type=str)
parser.add_argument("-m", "--masks", help="Output masks", required=True, type=bool)
parser.add_argument("--resolution", help="Resolution of the video", default=64, type=int)
parser.add_argument("--n_shapes", help="Maximum number of shapes", default=3, type=int)
parser.add_argument("--n_videos", help="Number of videos to generate", default=100, type=int)
parser.add_argument("--video_length", help="Length of the video", default=1000, type=int)
parser.add_argument("--background", help="Background color", default="black", type=str)

args = parser.parse_args()

object_colors = ImageColor.colormap
if args.background in object_colors:
    object_colors.pop(args.background)

video_generator = generate_videos(
    resolution=args.resolution,
    shapes=[Circle, Rectangle, Triangle],
    n_shapes=args.n_shapes,
    colors=list(object_colors.keys()),
    background=args.background,
    video_length=args.video_length,
)


