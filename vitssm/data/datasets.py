import os
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, Optional, Union

import gdown
import torch
from einops import rearrange
from torchvision.datasets import VisionDataset
from torchvision.io import read_video
from torchvision.tv_tensors import Video


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

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
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
        frame_skip: int = 1,
        num_frames: int = 10,
        output_format: Literal["TCHW", "THWC"] = "THWC",
        return_y: bool = False,
    ) -> None:
        super().__init__(root, transform=transform)

        self.train = train
        self.fold = fold
        self.frame_skip = frame_skip
        self.num_frames = num_frames
        self.output_format = output_format
        self.return_y = return_y
        
        assert output_format in ["TCHW", "THWC"], "Output format should be either 'TCWH' or 'THWC'."
        assert frame_skip > 0, "Frame skip should be greater than 0."
        assert 100 >= num_frames > 0, "Number of frames not in range 1-100."
        assert num_frames % frame_skip == 0, "Number of frames should be divisible by frame skip."

        if download and not Path(root).exists():
            self._download()

        folds_path = Path(root, "folds")
        folds_path = folds_path / f"train_{fold}.txt" if train else folds_path / f"test_{fold}.txt"
        with open(folds_path) as f:
            self.samples = f.read().splitlines()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Any:
        video_path = self.samples[idx]
        video = Video(
            read_video(
                video_path,
                pts_unit="sec",
                output_format="TCHW",
            )[0],
        )[0 : self.num_frames*self.frame_skip : self.frame_skip]

        if self.transform is not None:
            video = self.transform(video)

        video = rearrange(video, f"T C H W -> {" ".join(self.output_format)}")
        return video, video if self.return_y else video

    def _download(self) -> None:
        file_id = "1T7Vq7a70hu5949FJMe8yie5tvAwOmxXV"
        download_url = f"https://drive.google.com/uc?id={file_id}"
        output_path = str(self.root) + ".zip"

        gdown.download(download_url, output_path, quiet=False)

        with zipfile.ZipFile(output_path, "r") as zip_ref:
            zip_ref.extractall(os.path.dirname(self.root))

        os.remove(output_path)
