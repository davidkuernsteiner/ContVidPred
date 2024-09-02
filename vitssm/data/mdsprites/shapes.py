from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Union, Generator

from PIL import ImageDraw


@dataclass
class ShapeConfig:
    color: Tuple[int, int, int]
    position: Tuple[float, float]
    size: Tuple[int, int]
    rotation: int


class Shape(ABC):
    def __init__(self, config: ShapeConfig):
        self.position = config.position
        self.size = config.size
        self.rotation = config.rotation
        self.color = config.color

    @abstractmethod
    def draw(self, img: ImageDraw):
        pass


class Circle(Shape):
    def __init__(self, config: ShapeConfig):
        super().__init__(config)

    def draw(self, img: ImageDraw):
        img.ellipse(
            (
                self.position[0] - self.size[0] / 2,
                self.position[1] - self.size[1] / 2,
                self.position[0] + self.size[0] / 2,
                self.position[1] + self.size[1] / 2,
            ),
            fill=self.color,
        )


class Triangle(Shape):
    def __init__(self, config: ShapeConfig):
        super().__init__(config)

    def draw(self, img: ImageDraw):
        points = [
            (self.position[0] - self.size[0] / 2, self.position[1] + self.size[1] / 2),
            (self.position[0] + self.size[0] / 2, self.position[1] + self.size[1] / 2),
            (self.position[0], self.position[1] - self.size[1] / 2),
        ]
        img.polygon(points, fill=self.color)


class Rectangle(Shape):
    def __init__(self, config: ShapeConfig):
        super().__init__(config)

    def draw(self, img: ImageDraw):
        img.rectangle(
            (
                self.position[0] - self.size[0] / 2,
                self.position[1] - self.size[1] / 2,
                self.position[0] + self.size[0] / 2,
                self.position[1] + self.size[1] / 2,
            ),
            fill=self.color,
        )


class Heart(Shape):
    def __init__(self, config: ShapeConfig):
        super().__init__(config)

    def draw(self, img: ImageDraw):
        polygon = [
            (self.size[1] / 10, self.size[0] / 3),
            (self.size[1] / 10, 81 * self.size[0] / 120),
            (self.size[1] / 2, self.size[0]),
            (self.size[1] - self.size[1] / 10, 81 * self.size[0] / 120),
            (self.size[1] - self.size[1] / 10, self.size[0] / 3),
        ]
        img.polygon(polygon, fill=self.color)

        img.ellipse((0, 0, self.size[1] / 2, 3 * self.size[0] / 4), fill=self.color)
        img.ellipse((self.size[1] / 2, 0, self.size[1], 3 * self.size[0] / 4), fill=self.color)
