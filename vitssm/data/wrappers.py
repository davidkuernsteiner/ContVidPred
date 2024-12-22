import random

import torch
from einops import rearrange
from torchvision.transforms.v2 import InterpolationMode, Normalize, Resize
import random
import math
from torchvision.transforms import InterpolationMode

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .utils import make_coord



#TAKEN FROM: https://github.com/2y7c3/Super-Resolution-Neural-Operator/blob/main/datasets/wrappers.py
def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, InterpolationMode.BILINEAR)(
            transforms.ToPILImage()(img)))
    

class VariableResolutionAEDatasetWrapper:  
        def __init__(
            self,
            dataset: torch.utils.data.Dataset,
            res_x: int,
            train: bool,
            max_rescale_factor: int,
            min_rescale_factor: int = 1,
        ) -> None:
            super().__init__()

            self.dataset = dataset
            self.res_x = res_x
            self.train = train
            self.max_rescale_factor = max_rescale_factor
            self.min_rescale_factor = min_rescale_factor
            self.normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=False)

        def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor] | torch.Tensor]:
            outputs = self.dataset[index]
            if self.train:
                rescale_factor = random.uniform(self.min_rescale_factor, self.max_rescale_factor)
            else:
                rescale_factor = self.max_rescale_factor
  
            x = self.normalize(resize_fn(outputs, self.res_x))
            
            if outputs.size(-2) == round(self.res_x * rescale_factor):
                y = outputs
            else:
                y = resize_fn(outputs, round(self.res_x * rescale_factor))
            
            if self.train:
                y = self.normalize(y)
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
        

class VariableResolutionNextFrameDatasetWrapper:
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        res_x: int,
        context_length: int = 1,
    ) -> None:
        super().__init__()

        self.dataset = dataset
        self.res_x = res_x
        self.context_length = context_length
        
        self.normalize_x = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=False)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        frames = self.dataset[index]
        x, y = frames[:self.context_length], frames[self.context_length:]
        
        x = resize_fn(x, self.res_x)
        x = self.normalize_x(x)
        
        return x, y

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
        

#ADAPTED FROM: https://github.com/2y7c3/Super-Resolution-Neural-Operator/blob/main/datasets/wrappers.py
class SRImplicitDownsampledFast(Dataset):

    def __init__(
        self,
        dataset,
        inp_size: int = 32,
        scale_min: int = 1,
        scale_max: int = 4,
        train: bool = False,
    ):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.train = train
        
        self.normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        img = self.dataset[idx]
        s = random.uniform(self.scale_min, self.scale_max)

        h_lr, w_lr = self.inp_size, self.inp_size
        h_hr, w_hr = round(h_lr * s), round(w_lr * s)

        img_hr = resize_fn(img, (h_hr, w_hr))
        img_lr = resize_fn(img, (h_lr, w_lr))
        
        self.normalize(img_hr)
        self.normalize(img_lr)

        if self.train:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            img_lr = augment(img_lr)
            img_hr = augment(img_hr)

        hr_coord = make_coord([h_hr, w_hr], flatten=False)
        hr_rgb = img_hr

        if self.train:
            
            idcs = torch.tensor(np.random.choice(h_hr*w_hr, h_lr*w_lr, replace=False))

            hr_coord = hr_coord.view(-1, hr_coord.shape[-1])
            hr_coord = hr_coord[idcs, :]
            hr_coord = hr_coord.view(h_lr, w_lr, hr_coord.shape[-1])

            hr_rgb = img_hr.contiguous().view(img_hr.shape[0], -1)
            hr_rgb = hr_rgb[:, idcs]
            hr_rgb = hr_rgb.view(img_hr.shape[0], h_lr, w_lr)
        
        cell = torch.tensor([2 / img_hr.shape[-2], 2 / img_hr.shape[-1]], dtype=torch.float32)
        
        return {
            'inp': img_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }    
