import os
import argparse
from datetime import datetime
from dotenv import load_dotenv
from omegaconf import OmegaConf, DictConfig
from pathlib import Path
from typing import Union

import wandb

load_dotenv()

from vitssm.data import get_dataloaders
from vitssm.models import build_model
from vitssm.engine.tasks import NextFrameEngine
from vitssm.utils import flatten_config


def main():
    base_config = OmegaConf.load(Path(os.environ['CONFIG_DIR']) / "base_config.yml")
    
    #run = wandb.init(
    #    job_type="train",
    #    entity=os.environ["WANDB_ENTITY"],
    #    project=base_config.experiment.project,
    #    group=base_config.experiment.group,
    #    name=base_config.experiment.name,
    #    id=base_config.experiment.name + "_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
    #    config=dict(base_config),
    #    resume="allow",
    #)
    
    model = build_model(base_config)
    engine = NextFrameEngine(model=model, run_object=base_config)
    
    #if run.resumed:
    #    engine._resume_checkpoint()
    
    train_loader, val_loader = get_dataloaders(base_config)
    
    engine.train(train_loader, val_loader)
    engine.run.finish()


if __name__ == '__main__':
    main()
