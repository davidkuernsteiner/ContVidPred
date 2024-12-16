import os
from pathlib import Path
from typing import Tuple, Union, Any

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import HMDB51, UCF101, Kinetics, MovingMNIST, VisionDataset
from torchvision.transforms.v2 import Compose, Normalize, Resize, ToDtype, InterpolationMode

from .datasets import *
from .utils import get_transforms_video


def get_transform(
    config: DictConfig,
) -> Compose:
    """Builds image transformations."""
    transform = Compose(
        [
            Resize(config.resolution, interpolation=InterpolationMode.BICUBIC),
            ToDtype(torch.float32, scale=True),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ],
    )

    return transform


def get_dataset(config: DictConfig) -> Any:
    """Builds dataset given config."""
    data_root = Path(os.environ["DATA_DIR"])

    match config.name:
        case "ucf101":
            res = config.get("resolution", 64)
            
            return AEDatasetWrapper(
                UCF101(
                    root=data_root / "UCF-101",
                    annotation_path=str(data_root / "splits" / "ucfTrainTestlist"),
                    frames_per_clip=config.get("frames_per_clip", 16),
                    step_between_clips=config.get("step_between_clips", 1),
                    train=config.get("mode", "train") == "train",
                    transform=get_transforms_video(image_size=(res, res)),
                    num_workers=config.get("num_workers", 1),
                    output_format="TCHW",
                ),
                return_y=config.get("return_y", True),
            )
            
        case "mdsprites-ae":
            fold = f"train_{config.get('fold', 0)}.csv" if config.get("mode", "train") == "train" else f"test_{config.get('fold', 0)}.csv"
            res = config.get("resolution", 64)
            
            return AEDatasetWrapper(
                ImageDataset(
                    data_path=str(data_root / "MDsprites" / "folds" / fold),
                    image_size=(res, res),
                    transform_name="center",
                )
            )
            
        case "vmdsprites-ae":
            fold = f"train_{config.get('fold', 0)}.csv" if config.get("mode", "train") == "train" else f"test_{config.get('fold', 0)}.csv"
            res = config.get("resolution", 64)
            
            if config.get("load_in_memory", True):
                return AEDatasetWrapper(
                    MemoryVideoDataset(
                        data_path=str(data_root / "VMDsprites" / "folds" / fold),
                        num_frames=config.get("num_frames", 10),
                        frame_interval=config.get("frame_interval", 1),
                        image_size=(res, res),
                        transform_name=config.get("transform_name", "center"),
                    ),
                    return_y=True,
                )
            else:
                return AEDatasetWrapper(
                    VideoDataset(
                        data_path=str(data_root / "VMDsprites" / "folds" / fold),
                        num_frames=config.get("num_frames", 10),
                        frame_interval=config.get("frame_interval", 1),
                        image_size=(res, res),
                        transform_name=config.get("transform_name", "center"),
                    ),
                    return_y=True,
                )
                
        case "vmdsprites-var-res-ae":
            fold = f"train_{config.get('fold', 0)}.csv" if config.get("mode", "train") == "train" else f"test_{config.get('fold', 0)}.csv"
            res = config.get("resolution", 128)
            
            if config.get("load_in_memory", True):
                return VariableResolutionAEDatasetWrapper(
                    MemoryVideoDataset(
                        data_path=str(data_root / "VMDsprites_128" / "folds" / fold),
                        num_frames=config.get("num_frames", 10),
                        frame_interval=config.get("frame_interval", 1),
                        image_size=(res, res),
                        transform_name="cont_center",
                    ),
                    res_x=config.get("resolution_x", 32),
                    train=config.get("mode", "train") == "train",
                    max_rescale_factor=config.get("max_rescale_factor", 4),
                )
            else:
                return VariableResolutionAEDatasetWrapper(
                    VideoDataset(
                        data_path=str(data_root / "VMDsprites_128" / "folds" / fold),
                        num_frames=config.get("num_frames", 10),
                        frame_interval=config.get("frame_interval", 1),
                        image_size=(res, res),
                        transform_name="cont_center",
                    ),
                    res_x=config.get("resolution_x", 32),
                    train=config.get("mode", "train") == "train",
                    max_rescale_factor=config.get("max_rescale_factor", 4),
                )

        case "vmdsprites-nextframe":
            fold = f"train_{config.get('fold', 0)}.csv" if config.get("mode", "train") == "train" else f"test_{config.get('fold', 0)}.csv"
            res = config.get("resolution", 64)
            
            if config.get("load_in_memory", True):
                return NextFrameDatasetWrapper(
                    MemoryVideoDataset(
                        data_path=str(data_root / "VMDsprites" / "folds" / fold),
                        num_frames=config.get("num_frames", 10),
                        frame_interval=config.get("frame_interval", 1),
                        image_size=(res, res),
                        transform_name=config.get("transform_name", "center"),
                    ),
                    context_length=config.get("context_length", 1),
                )
            else:
                return NextFrameDatasetWrapper(
                    VideoDataset(
                        data_path=str(data_root / "VMDsprites" / "folds" / fold),
                        num_frames=config.get("num_frames", 10),
                        frame_interval=config.get("frame_interval", 1),
                        image_size=(res, res),
                        transform_name=config.get("transform_name", "center"),
                    ),
                    context_length=config.get("context_length", 1),
                )
                
        case "vmdsprites-var-res-nextframe":
            fold = f"train_{config.get('fold', 0)}.csv" if config.get("mode", "train") == "train" else f"test_{config.get('fold', 0)}.csv"
            res = config.get("resolution", 64)
            
            if config.get("load_in_memory", True):
                return VariableResolutionNextFrameDatasetWrapper(
                    MemoryVideoDataset(
                        data_path=str(data_root / "VMDsprites" / "folds" / fold),
                        num_frames=config.get("num_frames", 10),
                        frame_interval=config.get("frame_interval", 1),
                        image_size=(res, res),
                        transform_name=config.get("transform_name", "center"),
                    ),
                    res_x=config.get("resolution_x", 32),
                    context_length=config.get("context_length", 1),
                )
            else:
                return VariableResolutionNextFrameDatasetWrapper(
                    VideoDataset(
                        data_path=str(data_root / "VMDsprites" / "folds" / fold),
                        num_frames=config.get("num_frames", 10),
                        frame_interval=config.get("frame_interval", 1),
                        image_size=(res, res),
                        transform_name=config.get("transform_name", "center"),
                    ),
                    res_x=config.get("resolution_x", 32),
                    context_length=config.get("context_length", 1),
                )

        case _:
            raise ValueError(f"Unknown dataset: {config.name}")


def get_dataloaders(
    config: DictConfig,
) -> Union[DataLoader, tuple[DataLoader, DataLoader]]:
    """Builds dataloaders for training and validation."""
    dataset = get_dataset(config)

    if config.mode == "train":
        train_set, val_set = random_split(
            dataset, [config.train_percentage, 1 - config.train_percentage],
        )

        train_loader = DataLoader(
            train_set,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.get("num_workers", 1),
            pin_memory=config.get("pin_memory", False),
            persistent_workers=config.get("persistent_workers", False),
        )
        val_loader = DataLoader(
            val_set,
            batch_size=config.get("val_batch_size", config.batch_size),
            shuffle=False,
            num_workers=config.get("num_workers", 1),
            pin_memory=config.get("pin_memory", False),
            persistent_workers=config.get("persistent_workers", False),
        )

        return train_loader, val_loader

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.get("num_workers", 1),
        pin_memory=config.get("pin_memory", False),
        persistent_workers=config.get("persistent_workers", False),
    )


def get_dataloaders_next_frame(
    config: DictConfig,
) -> tuple[DataLoader, DataLoader]:
    train_set = get_dataset(config)
    
    eval_config = config.copy()
    eval_config["num_frames"] = eval_config.get("rollout_length", 10) + eval_config.get("context_length", 1)
    eval_config["mode"] = "test"
    eval_set = get_dataset(eval_config)
    
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.get("num_workers", 1),
        pin_memory=config.get("pin_memory", False),
        persistent_workers=config.get("persistent_workers", False),
    )
    
    eval_loader = DataLoader(
        eval_set,
        batch_size=config.get("val_batch_size", config.batch_size),
        shuffle=False,
        num_workers=config.get("num_workers", 1),
        pin_memory=config.get("pin_memory", False),
        persistent_workers=config.get("persistent_workers", False),
    )
    
    return train_loader, eval_loader


def get_dataloaders_continuous_ae(
    config: DictConfig,
) -> tuple[DataLoader, DataLoader]:
    train_set = get_dataset(config)
    
    eval_config = config.copy()
    eval_config["mode"] = "test"
    eval_set = get_dataset(eval_config)
    
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.get("num_workers", 1),
        pin_memory=config.get("pin_memory", False),
        persistent_workers=config.get("persistent_workers", False),
    )
    
    eval_loader = DataLoader(
        eval_set,
        batch_size=config.get("val_batch_size", config.batch_size),
        shuffle=False,
        num_workers=config.get("num_workers", 1),
        pin_memory=config.get("pin_memory", False),
        persistent_workers=config.get("persistent_workers", False),
    )
    
    return train_loader, eval_loader


def get_dataloaders_continuous_next_frame(
    config: DictConfig,
) -> tuple[DataLoader, DataLoader]:
    
    eval_config = config.copy()
    
    config["resolution"] = config["resolution_x"]
    eval_config["num_frames"] = eval_config.get("rollout_length", 10) + eval_config.get("context_length", 1)
    eval_config["mode"] = "test"
    eval_config["name"] = "vmdsprites-var-res-nextframe"
    
    train_set = get_dataset(config)
    eval_set = get_dataset(eval_config)
    
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.get("num_workers", 1),
        pin_memory=config.get("pin_memory", False),
        persistent_workers=config.get("persistent_workers", False),
    )
    
    eval_loader = DataLoader(
        eval_set,
        batch_size=config.get("val_batch_size", config.batch_size),
        shuffle=False,
        num_workers=config.get("num_workers", 1),
        pin_memory=config.get("pin_memory", False),
        persistent_workers=config.get("persistent_workers", False),
    )
    
    return train_loader, eval_loader
    