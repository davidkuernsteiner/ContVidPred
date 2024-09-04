from typing import Optional, Callable, Tuple, Union, Any
from pathlib import Path

import torch
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import find_classes, make_dataset
from torchvision.io import read_video


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
        return frames[: -self.frame_offset], frames[self.frame_offset :]

    def __len__(self) -> int:
        return len(self.dataset)


class VideoMDSpritesDataset(VisionDataset):

    def __init__(
        self,
        root: Union[Path, str],
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform, target_transform)

        extensions = ("avi",)
        #self.fold = fold
        self.train = train

        if download:
            pass
        self.classes, class_to_idx = find_classes(self.root)
        self.samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file=None)


    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Any:
        video_path, label = self.samples[idx]
        return read_video(video_path), label