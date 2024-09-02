import numpy as np
from PIL import Image, ImageDraw
from matplotlib.colors import to_rgb

from .shapes import Shape


class Frame:
    def __init__(
        self,
        size: tuple[int, int],
        background_color: str,
        shapes: list[Shape]
    ) -> None:
        self.h = size[0]
        self.w = size[1]
        self.background_color = tuple(255 * i for i in to_rgb(background_color))
        self.shapes = shapes

    def draw(self):
        img = Image.fromarray(np.full((self.h, self.w, 3), self.background_color, dtype=np.uint8))
        mask_full = np.zeros((self.h, self.w), dtype=np.uint8)
        draw = ImageDraw.Draw(img)
        for i, shape in enumerate(self.shapes):
            shape.draw(draw)
            mask = Image.fromarray(np.zeros((self.h, self.w), dtype=np.uint8))
            mask_draw = ImageDraw.Draw(mask)
            shape.color = i + 1
            shape.draw(mask_draw)
            mask_full[np.array(mask) != 0] = i + 1

        masks = [Image.fromarray(255 * np.array(mask_full == i, dtype=np.uint8)) for i in range(len(self.shapes) + 1)]

        return img, masks


class Video(Frame):

    def __init__(
        self,
        res: tuple[int, int],
        background_color: str,
        shapes: list[Shape],
        n_frames: int = 1000,
    ) -> None:
        super().__init__(res, background_color, shapes)
        self.n_frames = n_frames
        self.velocities = np.random.uniform(-5, 5, (len(self.shapes), 2))

    def make(self):
        frames, masks = [], [[] for _ in range(len(self.shapes))]
        for i, frame in enumerate(range(self.n_frames)):
            for j, shape in enumerate(self.shapes):
                shape.position += self.velocities[j]
                if (shape.position[0] > self.h - shape.size[0]) or (shape.position[0] < 0 + shape.size[0]):
                    self.velocities[j][0] = -self.velocities[j][0]
                elif (shape.position[1] > self.w - shape.size[1]) or (shape.position[1] < 0 + shape.size[1]):
                    self.velocities[j][1] = -self.velocities[j][1]
            res = self.draw()
            frames.append(np.array(res[0]))

            for j in range(len(self.shapes)):
                masks[j].append(np.array(res[1][j]))

        return frames, masks
