import os
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, Optional, Union

import gdown
import torch
from einops import rearrange
from torchvision.datasets import VisionDataset
from torchvision.datasets.video_utils import VideoClips
from torchvision.io import read_video
from torchvision.tv_tensors import Video
from glob import glob

import numpy as np
from PIL import ImageFile
from torchvision.datasets.folder import IMG_EXTENSIONS, pil_loader


from .read import read_video
from .utils import VID_EXTENSIONS, get_transforms_image, get_transforms_video, read_file, temporal_random_crop


class AEDatasetWrapper:
    
        def __init__(
            self,
            dataset: VisionDataset,
            return_y: bool = False,
        ) -> None:
            super().__init__()
    
            self.dataset = dataset
            self.return_y = return_y
    
        def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
            outputs = self.dataset[index]
            return outputs[0], outputs[0] if self.return_y else outputs[0]
    
        def __len__(self) -> int:
            return len(self.dataset)
        

class NextFrameDatasetWrapper:

    def __init__(
        self,
        dataset: VisionDataset,
    ) -> None:
        super().__init__()

        self.dataset = dataset

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        frames = self.dataset[index]
        return frames[:-1], frames[-1].unsqueeze(0)

    def __len__(self) -> int:
        return len(self.dataset)
        

#ADAPTED FROM: https://github.com/hpcaitech/Open-Sora/blob/main/opensora/datasets/datasets.py
class VideoDataset(torch.utils.data.Dataset):
    """load video according to the csv file.

    Args:
        target_video_len (int): the number of video frames will be load.
        align_transform (callable): Align different videos in a specified size.
        temporal_sample (callable): Sample the target length of a video.
    """
    def __init__(
        self,
        data_path: str | None = None,
        num_frames: int = 16,
        frame_interval: int = 1,
        image_size: tuple[int, int] = (256, 256),
        transform_name: str = "center",
    ):
        self.data_path = data_path
        self.data = read_file(data_path)
        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.image_size = image_size
        self.transforms = get_transforms_video(transform_name, image_size),

    def getitem(self, index: int):
        sample = self.data.iloc[index]
        path = sample["path"]
        # loading
        vframes, vinfo = read_video(path, backend="av")
        video_fps = vinfo["video_fps"] if "video_fps" in vinfo else 24
        # Sampling video frames
        video = temporal_random_crop(vframes, self.num_frames, self.frame_interval)
        # transform
        video = self.transforms(video)  # T C H W
        # TCHW -> CTHW
        #video = video.permute(1, 0, 2, 3)

        return video

    def __getitem__(self, index: int):
        for _ in range(10):
            try:
                return self.getitem(index)
            except Exception as e:
                path = self.data.iloc[index]["path"]
                print(f"data {path}: {e}")
                index = np.random.randint(len(self))
        raise RuntimeError("Too many bad data.")

    def __len__(self):
        return len(self.data)
    
    
#ADAPTED FROM: https://github.com/hpcaitech/Open-Sora/blob/main/opensora/datasets/datasets.py
class ImageDataset(torch.utils.data.Dataset):
    """load video according to the csv file.

    Args:
        target_video_len (int): the number of video frames will be load.
        align_transform (callable): Align different videos in a specified size.
        temporal_sample (callable): Sample the target length of a video.
    """
    def __init__(
        self,
        data_path: str | None = None,
        image_size: tuple[int, int] = (256, 256),
        transform_name: str = "center",
    ):
        self.data_path = data_path
        self.data = read_file(data_path)
        self.image_size = image_size
        self.transforms = get_transforms_image(transform_name, image_size),

    def getitem(self, index: int):
        sample = self.data.iloc[index]
        path = sample["path"]
        # loading
        image = pil_loader(path)
        # transform
        image = self.transforms(image)

        return image

    def __getitem__(self, index: int):
        for _ in range(10):
            try:
                return self.getitem(index)
            except Exception as e:
                path = self.data.iloc[index]["path"]
                print(f"data {path}: {e}")
                index = np.random.randint(len(self))
        raise RuntimeError("Too many bad data.")

    def __len__(self):
        return len(self.data)


class VideoMDSpritesDataset(VisionDataset):
    #TODO change data loading to fix multiprocess error. see: https://towardsdatascience.com/ml-engineering-101-a-thorough-explanation-of-the-error-dataloader-worker-pid-s-xxx-exited-f3a6a983911e
    def __init__(
        self,
        root: Union[Path, str],
        download: bool = True,
        train: bool = True,
        fold: int = 0,
        transform: Optional[Callable] = None,
        frames_per_clip: int = 10,
        steps_between_clips: int = 1,
        output_format: Literal["TCHW", "THWC"] = "TCHW",
        return_y: bool = False,
    ) -> None:
        super().__init__(root, transform=transform)

        self.train = train
        self.fold = fold
        self.output_format = output_format
        self.return_y = return_y
        
        assert output_format in ["TCHW", "THWC"], "Output format should be either 'TCWH' or 'THWC'."

        if download and not Path(root).exists():
            self._download()

        folds_path = Path(root, "folds")
        folds_path = folds_path / f"train_{fold}.txt" if train else folds_path / f"test_{fold}.txt"
        with open(folds_path) as f:
            self.video_paths = f.read().splitlines()
            
        self.video_clips = VideoClips(
            self.video_paths,
            frames_per_clip,
            steps_between_clips,
            num_workers=4,
            output_format=output_format,
        )

    def __len__(self) -> int:
        return len(self.video_clips)

    def __getitem__(self, idx: int) -> Any:
        video, audio, info, video_idx = self.video_clips.get_clip(idx)

        if self.transform is not None:
            video = self.transform(video)

        return video

    def _download(self) -> None:
        file_id = "1T7Vq7a70hu5949FJMe8yie5tvAwOmxXV"
        download_url = f"https://drive.google.com/uc?id={file_id}"
        output_path = str(self.root) + ".zip"

        gdown.download(download_url, output_path, quiet=False)

        with zipfile.ZipFile(output_path, "r") as zip_ref:
            zip_ref.extractall(os.path.dirname(self.root))

        os.remove(output_path)