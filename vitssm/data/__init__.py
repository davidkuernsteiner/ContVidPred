from pathlib import Path
from typing import Tuple, Union

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import HMDB51, UCF101, Kinetics, MovingMNIST, VisionDataset
from torchvision.transforms.v2 import Compose, Normalize, Resize, ToDtype

from .datasets import NextFrameDataset, VideoMDSpritesDataset


def get_transforms(
    config: DictConfig,
) -> Compose:
    """Builds image transformations."""
    transform = Compose(
        [
            Resize(config.dataset.resolution),
            ToDtype(torch.float32, scale=True),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ],
    )

    return transform


def get_dataset(config: DictConfig) -> Union[VisionDataset, NextFrameDataset]:
    """Builds dataset given config."""
    data_root = Path(config.root_dir, config.dataset.get("root", "data"))

    match config.dataset.name:
        case "moving_mnist":
            return NextFrameDataset(
                MovingMNIST(
                    root=data_root,
                    split=config.dataset.get("mode", "train"),
                    download=True,
                    transform=get_transforms(config),
                ),
            )

        case "kinetics":
            return Kinetics(
                root=data_root / "kinetics",
                split=config.dataset.get("mode", "train"),
                download=True,
                transform=get_transforms(config),
                output_format="TCHW",
            )

        case "ucf101":
            return UCF101(
                root=data_root / "UCF-101",
                annotation_path=data_root / "splits" / "ucfTrainTestlist",
                frames_per_clip=config.dataset.get("frames_per_clip", 16),
                step_between_clips=config.dataset.get("step_between_clips", 1),
                train=config.dataset.get("mode", "train") == "train",
                transform=get_transforms(config),
                output_format="TCHW",
            )

        case "hmdb51":
            return HMDB51(
                root=data_root / "HMDB51",
                annotation_path=data_root / "splits" / "testTrainMulti_7030_splits",
                frames_per_clip=config.dataset.get("frames_per_clip", 16),
                train=config.dataset.get("mode", "train") == "train",
                transform=get_transforms(config),
                output_format="TCHW",
            )

        case "vmdsprites":
            return NextFrameDataset(
                VideoMDSpritesDataset(
                    root=data_root / "VMDsprites",
                    train=config.dataset.get("mode", "train") == "train",
                    fold=config.dataset.get("fold", 0),
                    transform=get_transforms(config),
                ),
                frame_offset=config.dataset.get("frame_offset", 1),
            )

        case _:
            raise ValueError(f"Unknown dataset: {config.dataset.name}")


def get_dataloaders(
    config: DictConfig,
) -> Union[DataLoader, tuple[DataLoader, DataLoader]]:
    """Builds dataloaders for training and validation."""
    dataset = get_dataset(config)

    if config.dataset.mode == "train":
        train_set, val_set = random_split(
            dataset, [config.dataset.train_percentage, 1 - config.dataset.train_percentage],
        )

        train_loader = DataLoader(
            train_set,
            batch_size=config.optimization.batch_size,
            shuffle=True,
            num_workers=config.dataset.get("num_workers", 1),
            pin_memory=config.dataset.get("pin_memory", False),
        )
        val_loader = DataLoader(
            val_set,
            batch_size=config.optimization.get("val_batch_size", config.optimization.batch_size),
            shuffle=False,
            num_workers=config.dataset.get("num_workers", 1),
            pin_memory=config.dataset.get("pin_memory", False),
        )

        return train_loader, val_loader

    return DataLoader(
        dataset,
        batch_size=config.optimization.batch_size,
        shuffle=False,
        num_workers=config.get("num_workers", 1),
        pin_memory=config.dataset.get("pin_memory", False),
    )
