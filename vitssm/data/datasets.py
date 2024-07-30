from typing import Optional, Callable, Tuple, Union
from pathlib import Path

import torch
from torchvision.datasets import VisionDataset


class NextFrameDataset:

    def __init__(
        self,
        dataset: VisionDataset,
        frame_offset: int = 1,
    ) -> None:
        super().__init__()

        self.dataset = dataset
        self.frame_offset = frame_offset

        assert len(self.dataset[0]) > self.frame_offset, "Sample length should be greater than frame offset."

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        frames = self.dataset[index]
        return frames[:-self.frame_offset], frames[self.frame_offset:]

    def __len__(self) -> int:
        return len(self.dataset)



class VideoMDSpritesDataset(VisionDataset):

    def __init__(
        self,
        root: Union[Path, str],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,

    ):
        super().__init__(root, transform, target_transform)
