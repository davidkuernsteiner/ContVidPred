from typing import Union, Tuple, Any
import os

import torch
from addict import Dict
from torch.utils.data import DataLoader, random_split
from torchvision.transforms.v2 import Compose, Resize, ToImage, ToDtype, Normalize
from torchvision.datasets import VisionDataset, MovingMNIST, Kinetics

def build_transforms(
        config: Dict,
) -> Compose:
    """Builds image transformations."""
    transform = Compose([
        Resize(config.dataset.resolution),
        ToImage(),
        ToDtype(torch.float32, scale=True),
    ])

    return transform


def build_dataset(
        config: Dict
) -> VisionDataset:
    """Builds dataset given config."""
    data_root = os.path.join(config.dataset.root_dir, config.dataset.get("root", "data"))

    match config.dataset.name:
        case "moving_mnist":
            return MovingMNIST(root=data_root,
                               split=config.dataset.get("mode", "train"),
                               download=True,
                               transform=build_transforms(config)
                               )

        case "kinetics_400":
            return Kinetics(root=data_root,
                            split=config.dataset.get("mode", "train"),
                            download=True,
                            transform=build_transforms(config),
                            )

        case _:
            raise ValueError(f"Unknown dataset: {config.dataset.name}")



def build_dataloaders(
        config: Dict,
) -> Union[DataLoader, Tuple[DataLoader, DataLoader]]:
    """Builds dataloaders for training and validation."""
    dataset = build_dataset(config)
    if config.dataset.mode == "train":
        train_set, val_set = random_split(dataset,
                                          [config.dataset.train_percentage,
                                           1 - config.dataset.train_percentage]
                                          )
        train_loader = DataLoader(train_set,
                                  batch_size=config.optimization.batch_size,
                                  shuffle=True,
                                  num_workers=config.dataset.get("num_workers", 1),
                                  pin_memory=config.dataset.get("pin_memory", False)
                                  )
        val_loader = DataLoader(val_set,
                                batch_size=config.optimization.get("val_batch_size",
                                                                   config.optimization.batch_size),
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
