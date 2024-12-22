import os
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, Optional, Union

import gdown
import torch
from torchvision.datasets import VisionDataset
from torchvision.datasets.video_utils import VideoClips

import numpy as np
import cv2
from PIL import Image
from torchvision.datasets.folder import pil_loader


from .read import read_video
from .utils import get_transforms_image, get_transforms_video, read_file, temporal_random_crop
from . import transforms



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
        data_path: str,
        video_length: int,
        clip_length: int = 16,
        frame_interval: int = 1,
        image_size: tuple[int, int] = (32, 32),
        transform_name: str = "center",
    ):
        self.data_path = data_path
        self.data = read_file(data_path)
        
        self.video_length = video_length
        self.clip_length = clip_length
        self.frame_interval = frame_interval
        self.image_size = image_size
        
        self.transforms = get_transforms_video(transform_name, image_size)
        
        self.clips = []
        for path in self.data["path"]:
            clips = [(path, i, i+self.clip_length) for i in range(0, video_length - clip_length + 1, frame_interval)]
            self.clips.extend(clips)

    def getitem(self, index: int):
        path, idx_start, idx_end = self.clips[index]
        # loading
        vframes, _ = read_video(path, backend="cv2")
        # Sampling video frames
        video = vframes[idx_start:idx_end]
        # transform
        video = self.transforms(video)  # T C H W

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
        return len(self.clips)
    

class VideoFrameDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data_path: str,
            video_length: int,
            image_size: tuple[int, int] = (32, 32),
            transform_name: str | None = None,
        ):
        """
        Args:
            video_dir (string): Directory with all the video files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_path = data_path
        self.video_length = video_length
        self.paths = read_file(data_path)
        self.image_size = image_size
        self.transforms = get_transforms_image(transform_name, image_size) if transform_name else None
        self.frame_indices = self._get_all_frame_indices()

    def _get_all_frame_indices(self):
        frame_indices = []
        for video_path in self.paths["path"]:
            frame_indices.extend([(video_path, i) for i in range(self.video_length)])
        return frame_indices

    def __len__(self):
        return len(self.frame_indices)

    def __getitem__(self, idx):
        video_path, frame_idx = self.frame_indices[idx]
        vframes, _ = read_video(video_path, backend="cv2")
        frame = vframes[frame_idx]
        
        if self.transforms:
            frame = self.transforms(frame)
        
        return frame
    
    
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
        self.transforms = get_transforms_image(transform_name, image_size)

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
        folds_path = folds_path / f"train_{fold}.csv" if train else folds_path / f"test_{fold}.csv"
        self.video_paths = list(read_file(str(folds_path))["path"])
            
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