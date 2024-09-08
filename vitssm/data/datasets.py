from typing import Literal, Optional, Callable, Tuple, Union, Any
from pathlib import Path
import os

import torch
from torchvision.datasets import VisionDataset
from torchvision.io import read_video
import gdown
import zipfile


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
        download: bool = True,
        train: bool = True,
        fold: int = 0,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        output_format: Literal["TCWH", "THWC"] = "THWC",
    ) -> None:
        super().__init__(root, transform, target_transform)

        self.train = train
        self.fold = fold
        self.output_format = output_format
        
        if download and not Path(root).exists():
            self._download()
            

        folds_path = Path(root, "folds")
        folds_path = folds_path / f"train_{fold}.txt" if train else folds_path / f"test_{fold}.txt"
        with open(folds_path, "r") as f:
            self.samples = f.read().splitlines()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Any:
        video_path = self.samples[idx]
        return read_video(
            video_path,
            pts_unit="sec",
            output_format=self.output_format,
        )[0]
        
    def _download(self) -> None:
        file_id = "1T7Vq7a70hu5949FJMe8yie5tvAwOmxXV"
        download_url = f"https://drive.google.com/uc?id={file_id}"
        output_path = self.root + ".zip"

        gdown.download(download_url, output_path, quiet=False)

        with zipfile.ZipFile(output_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(self.root))
        
        os.remove(output_path)
