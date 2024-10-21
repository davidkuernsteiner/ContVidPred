from collections.abc import Generator
from typing import Union

import numpy as np

from .frame import Frame, Video
from .shapes import ShapeConfig


def generate_images(
    resolution: int,
    generate_masks: bool,
    shapes: list,
    n_shapes: Union[int, tuple[int, int]],
    colors: list,
    background: str,
) -> Generator:

    while True:
        n_ = n_shapes if isinstance(n_shapes, int) else np.random.randint(n_shapes[0], n_shapes[1] + 1)
        shapes_ = np.random.choice(shapes, n_)
        colors_ = np.random.choice(colors, n_)
        positions = np.random.uniform(resolution * 0.1, resolution - resolution * 0.1, (n_, 2))
        sizes = np.random.uniform(resolution * 0.1, resolution * 0.3, (n_, 2))
        shapes_ = [
            shape(ShapeConfig(color, pos, size, 0))
            for shape, color, pos, size in zip(shapes_, colors_, positions, sizes, strict=False)
        ]

        yield Frame((resolution, resolution), background, shapes_, masks=generate_masks).draw()


def generate_videos(
    resolution: int,
    generate_masks: bool,
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
        positions = np.random.uniform(resolution * 0.2, resolution - resolution * 0.2, (n_, 2))
        sizes = np.random.uniform(resolution * 0.1, resolution * 0.3, (n_, 2))
        shapes_ = [
            shape(ShapeConfig(color, pos, size, 0))
            for shape, color, pos, size in zip(shapes_, colors_, positions, sizes, strict=False)
        ]

        yield Video((resolution, resolution), background, shapes_, generate_masks, video_length).make()
