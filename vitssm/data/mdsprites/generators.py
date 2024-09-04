from typing import Union, Generator

import numpy as np

from .frame import Frame, Video
from .shapes import ShapeConfig
from PIL.ImageColor import getcolor


def generate_images(
    img_size, shapes: list, n_shapes: Union[int, tuple[int, int]], colors: list, background: str, n_images: int
) -> Generator:

    for _ in range(n_images):
        n_ = n_shapes if isinstance(n_shapes, int) else np.random.randint(n_shapes[0], n_shapes[1] + 1)
        shapes_ = np.random.choice(shapes, n_)
        # colors_ = np.random.choice(colors, n_)
        colors_ = [(r, g, b) for r, g, b in np.random.randint(0, 256, (n_, 3))]
        positions = np.random.uniform(img_size * 0.1, img_size - img_size * 0.1, (n_, 2))
        sizes = np.round(np.random.uniform(img_size * 0.1, img_size * 0.3, (n_, 2))).astype(np.int32)
        shapes_ = [
            shape(ShapeConfig(color, pos, size, 0))
            for shape, color, pos, size in zip(shapes_, colors_, positions, sizes)
        ]

        yield Frame((img_size, img_size), background, shapes_).draw()


def generate_videos(
    resolution: int,
    masks: bool,
    shapes: list,
    n_shapes: Union[int, tuple[int, int]],
    colors: list,
    background: str,
    video_length: int = 1000,
) -> Generator:

    while True:
        n_ = n_shapes if isinstance(n_shapes, int) else np.random.randint(n_shapes[0], n_shapes[1] + 1)
        shapes_ = np.random.choice(shapes, n_)
        colors_ = np.random.choice(colors, n_)
        # colors_ = np.random.randint(0, 256, (n_, 3))
        positions = np.random.uniform(resolution * 0.1, resolution - resolution * 0.1, (n_, 2))
        sizes = np.random.uniform(resolution * 0.1, resolution * 0.3, (n_, 2))
        shapes_ = [
            shape(ShapeConfig(color, pos, size, 0))
            for shape, color, pos, size in zip(shapes_, colors_, positions, sizes)
        ]

        yield Video((resolution, resolution), background, shapes_, video_length).make()
