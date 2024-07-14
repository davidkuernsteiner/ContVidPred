from typing import Union, Tuple, Any
import os
from pathlib import Path

import torch
from addict import Dict
from torch.utils.data import DataLoader, random_split
from torchvision.transforms.v2 import Compose, Resize, ToImage, ToDtype, Normalize
from torchvision.datasets import VisionDataset, MovingMNIST, Kinetics, UCF101, HMDB51
from omegaconf import DictConfig

def build_transforms(
        config: DictConfig,
) -> Compose:
    """Builds image transformations."""
    transform = Compose([
        Resize(config.dataset.resolution),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return transform


def build_dataset(
        config: DictConfig
) -> VisionDataset:
    """Builds dataset given config."""
    data_root = Path(config.root_dir, config.dataset.get("root", "data"))

    match config.dataset.name:
        case "moving_mnist":
            return MovingMNIST(
                root=data_root,
                split=config.dataset.get("mode", "train"),
                download=True,
                transform=build_transforms(config)
            )

        case "kinetics":
            return Kinetics(
                root=data_root / "kinetics",
                split=config.dataset.get("mode", "train"),
                download=True,
                transform=build_transforms(config),
                output_format="TCHW",
            )

        case "ucf101":
            return UCF101(
                root=data_root / "UCF-101",
                annotation_path= data_root / "splits" / "ucfTrainTestlist",
                frames_per_clip=config.dataset.get("frames_per_clip", 16),
                step_between_clips=config.dataset.get("step_between_clips", 1),
                train=config.dataset.get("mode", "train") == "train",
                transform=build_transforms(config),
                output_format="TCHW",
            )

        case "hmdb51":
            return HMDB51(
                root=data_root / "HMDB51",
                annotation_path= data_root / "splits" / "testTrainMulti_7030_splits",
                frames_per_clip=config.dataset.get("frames_per_clip", 16),
                train=config.dataset.get("mode", "train") == "train",
                transform=build_transforms(config),
                output_format="TCHW",
            )

        case _:
            raise ValueError(f"Unknown dataset: {config.dataset.name}")


def build_dataloaders(
        config: DictConfig,
) -> Union[DataLoader, Tuple[DataLoader, DataLoader]]:
    """Builds dataloaders for training and validation."""
    dataset = build_dataset(config)

    if config.dataset.mode == "train":
        train_set, val_set = random_split(
            dataset,
            [config.dataset.train_percentage, 1 - config.dataset.train_percentage]
        )

        train_loader = DataLoader(train_set,
                                  batch_size=config.optimization.batch_size,
                                  shuffle=True,
                                  num_workers=config.dataset.get("num_workers", 1),
                                  pin_memory=config.dataset.get("pin_memory", False)
                                  )
        val_loader = DataLoader(val_set,
                                batch_size=config.optimization.get(
                                    "val_batch_size",
                                    config.optimization.batch_size
                                ),
                                shuffle=False,
                                num_workers=config.dataset.get("num_workers", 1),
                                pin_memory=config.dataset.get("pin_memory", False)
                                )

        return train_loader, val_loader

    else:
        return DataLoader(dataset,
                          batch_size=config.optimization.batch_size,
                          shuffle=False,
                          num_workers=config.get("num_workers", 1),
                          pin_memory=config.dataset.get("pin_memory", False)
                          )
