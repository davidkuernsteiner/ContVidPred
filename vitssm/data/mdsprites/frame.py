from typing import Union

import numpy as np
from matplotlib.colors import to_rgb
from PIL import Image, ImageDraw

from .shapes import Shape


class Frame:
    def __init__(
        self,
        size: tuple[int, int],
        background_color: str,
        shapes: list[Shape],
        masks: bool = False,
    ) -> None:
        super().__init__()
        self.h = size[0]
        self.w = size[1]
        self.background_color = tuple(255 * i for i in to_rgb(background_color))
        self.shapes = shapes
        self.masks = masks

    def draw(self):
        img = Image.fromarray(np.full((self.h, self.w, 3), self.background_color, dtype=np.uint8))
        draw = ImageDraw.Draw(img)

        masks = []
        mask_full = np.zeros((self.h, self.w), dtype=np.uint8)
        for i, shape in enumerate(self.shapes):
            shape.draw(draw)
            if self.masks:
                mask = Image.fromarray(np.zeros((self.h, self.w), dtype=np.uint8))
                mask_draw = ImageDraw.Draw(mask)
                shape.color = i + 1
                shape.draw(mask_draw)
                mask_full[np.array(mask) != 0] = i + 1

        masks = (
            [Image.fromarray(255 * np.array(mask_full == i, dtype=np.uint8)) for i in range(len(self.shapes) + 1)]
            if self.masks
            else None
        )

        return img, masks


class Video(Frame):
    def __init__(
        self,
        resolution: tuple[int, int],
        background_color: str,
        shapes: list[Shape],
        masks: bool = False,
        n_frames: int = 100,
    ) -> None:
        super().__init__(resolution, background_color, shapes, masks)
        self.n_frames = n_frames
        self.velocities = np.random.uniform(-0.05 * resolution[0], 0.05 * resolution[0], (len(self.shapes), 2))
        self.velocity_lock = np.ones((len(self.shapes), 2), dtype=np.uint8) * 5

    def make(self):
        frames, masks = [], [[] for _ in range(len(self.shapes))] if self.masks else None
        for _ in range(self.n_frames):
            for j, shape in enumerate(self.shapes):
                shape.position += self.velocities[j]
                if (
                    (shape.position[0] > (self.h - shape.size[0] / 2)) or (shape.position[0] < (0 + shape.size[0] / 2))
                ) and (self.velocity_lock[j][0] >= 5):
                    self.velocities[j][0] = -self.velocities[j][0]
                    self.velocity_lock[j][0] = 0

                elif (
                    (shape.position[1] > (self.w - shape.size[1] / 2)) or (shape.position[1] < (0 + shape.size[1] / 2))
                ) and (self.velocity_lock[j][1] >= 5):
                    self.velocities[j][1] = -self.velocities[j][1]
                    self.velocity_lock[j][1] = 0
            self.velocity_lock += 1
            res = self.draw()
            frames.append(np.array(res[0]))

            if self.masks:
                for j in range(len(self.shapes)):
                    masks[j].append(np.array(res[1][j]))

        return np.stack(frames, axis=0), np.stack(np.stack(masks, axis=1), axis=0) if self.masks else None
