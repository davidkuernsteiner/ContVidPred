import os
import argparse
import yaml
from datetime import datetime
from dotenv import load_dotenv
from omegaconf import OmegaConf, DictConfig
from pathlib import Path
from typing import Union

import wandb
from wandb.sdk import launch

load_dotenv()

from vitssm.data import get_dataloaders
from vitssm.models import build_model
from vitssm.engine.tasks import NextFrameEngine
from vitssm.utils import flatten_config


wandb.login()
wandb.require("core")


def main():
    
    config_path = Path(os.environ["CONFIG_DIR"]) / "base_config.yml"
    
    base_config = OmegaConf.load(config_path)
    
    with wandb.init(
        job_type="train",
        group=base_config.group,
        name=base_config.name,
        config=dict(base_config),
        resume="allow",
    ):
        launch.manage_wandb_config(
        include=["dataset", "optimization", "metrics", "model"], 
        exclude=["project", "log_freq", "seed"],
        )

        run_config = DictConfig(launch.load_wandb_config())
        model = build_model(run_config)
        engine = NextFrameEngine(model=model, run_object=run_config)

        #if wandb.run.resumed:
        #    engine._resume_checkpoint()

        train_loader, val_loader = get_dataloaders(run_config)

        engine.train(train_loader, val_loader)


if __name__ == '__main__':
    main()
