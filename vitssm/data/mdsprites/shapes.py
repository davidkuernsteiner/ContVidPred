from abc import ABC, abstractmethod
from dataclasses import dataclass

from PIL.ImageDraw import ImageDraw


@dataclass
class ShapeConfig:
    color: tuple[int, int, int]
    position: tuple[float, float]
    size: tuple[int, int]
    rotation: int


class Shape(ABC):
    def __init__(self, config: ShapeConfig) -> None:
        super().__init__()
        self.position = config.position
        self.size = config.size
        self.rotation = config.rotation
        self.color = config.color

    @abstractmethod
    def draw(self, img: ImageDraw) -> None:
        pass


class Circle(Shape):
    def __init__(self, config: ShapeConfig) -> None:
        super().__init__(config)

    def draw(self, img: ImageDraw) -> None:
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
    def __init__(self, config: ShapeConfig) -> None:
        super().__init__(config)

    def draw(self, img: ImageDraw) -> None:
        points = [
            (self.position[0] - self.size[0] / 2, self.position[1] + self.size[1] / 2),
            (self.position[0] + self.size[0] / 2, self.position[1] + self.size[1] / 2),
            (self.position[0], self.position[1] - self.size[1] / 2),
        ]
        img.polygon(points, fill=self.color)


class Rectangle(Shape):
    def __init__(self, config: ShapeConfig) -> None:
        super().__init__(config)

    def draw(self, img: ImageDraw) -> None:
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
    def __init__(self, config: ShapeConfig) -> None:
        super().__init__(config)

    def draw(self, img: ImageDraw) -> None:
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
