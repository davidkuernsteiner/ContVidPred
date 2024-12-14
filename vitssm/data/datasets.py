import os
import zipfile
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, Optional, Union

import gdown
import torch
from einops import rearrange
from torchvision.datasets import VisionDataset
from torchvision.datasets.video_utils import VideoClips
from torchvision.tv_tensors import Video
import torchvision.transforms.v2.functional as F
from torchvision.transforms.v2 import InterpolationMode
from glob import glob

import numpy as np
from PIL import ImageFile
from torchvision.datasets.folder import IMG_EXTENSIONS, pil_loader


from .read import read_video
from .utils import VID_EXTENSIONS, get_transforms_image, get_transforms_video, read_file, temporal_random_crop


class VariableResolutionAEDatasetWrapper:  
        def __init__(
            self,
            dataset: torch.utils.data.Dataset,
            res_x: int,
            train: bool,
            max_rescale_factor: int,
        ) -> None:
            super().__init__()

            self.dataset = dataset
            self.res_x = res_x
            self.train = train
            self.max_rescale_factor = max_rescale_factor

        def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor] | torch.Tensor]:
            outputs = self.dataset[index]
            if self.train:
                rescale_factor = random.choice(list(range(1, self.max_rescale_factor + 1)))
            else:
                rescale_factor = self.max_rescale_factor
  
            x = F.resize(outputs, self.res_x, interpolation=InterpolationMode.BICUBIC)
            if outputs.size(-2) == self.res_x * rescale_factor:
                y = outputs
            else:
                y = F.resize(outputs, self.res_x * rescale_factor, interpolation=InterpolationMode.BICUBIC)
            
            if self.train:
                x_cl, _, x_ht, x_wt = x.shape
                y_cl, _, y_ht, y_wt = y.shape
                coords = rearrange(
                    torch.stack(
                        torch.meshgrid(
                            [
                                torch.arange(int(y_cl)),
                                torch.arange(int(y_ht)),
                                torch.arange(int(y_wt))
                            ], 
                            indexing="ij",
                        )
                    ),
                    "ndim time height width -> (time height width) ndim",
                ).float()
                dims = torch.tensor([int(y_cl), int(y_ht), int(y_wt)])
                coords = coords / (dims - 1) * 1000

                y_values = rearrange(y, "cl ch ht wt -> (cl ht wt) ch")

                subsample_idcs = torch.randperm(coords.size(0))[:x_cl*x_ht*x_wt]
                coords = coords[subsample_idcs]
                y_values = y_values[subsample_idcs]

                return x, {"coords": coords, "values": y_values}
            
            else:
                return x, y

        def __len__(self) -> int:
            return len(self.dataset)
        

class AEDatasetWrapper:  
        def __init__(
            self,
            dataset: torch.utils.data.Dataset,
            return_y: bool = False,
        ) -> None:
            super().__init__()
    
            self.dataset = dataset
            self.return_y = return_y
    
        def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
            outputs = self.dataset[index]
            return outputs, outputs if self.return_y else outputs
    
        def __len__(self) -> int:
            return len(self.dataset)
        

class NextFrameDatasetWrapper:
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        context_length: int = 1,
    ) -> None:
        super().__init__()

        self.dataset = dataset
        self.context_length = context_length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        frames = self.dataset[index]
        return frames[:self.context_length], frames[self.context_length:]

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
        image_size: tuple[int, int] = (32, 32),
        transform_name: str = "center",
    ):
        self.data_path = data_path
        self.data = read_file(data_path)
        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.image_size = image_size
        self.transforms = get_transforms_video(transform_name, image_size)

    def getitem(self, index: int):
        sample = self.data.iloc[index]
        path = sample["path"]
        # loading
        vframes, _ = read_video(path, backend="cv2")
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
    
    
class MemoryVideoDataset(torch.utils.data.Dataset):
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
        image_size: tuple[int, int] = (32, 32),
        transform_name: str = "center",
    ):
        self.data_path = data_path
        self.paths = read_file(data_path)
        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.image_size = image_size
        self.transforms = get_transforms_video(transform_name, image_size)
        
        self.data = []
        for path in self.paths["path"]:
            vframes, _ = read_video(path, backend="cv2")
            self.data.append(vframes)
            

    def __getitem__(self, index: int):
        sample = self.data[index]
        # Sampling video frames
        video = temporal_random_crop(sample, self.num_frames, self.frame_interval)
        # transform
        video = self.transforms(video)  # T C H W
        # TCHW -> CTHW
        #video = video.permute(1, 0, 2, 3)

        return video

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