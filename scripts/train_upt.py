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
from torch.utils.data import DataLoader

load_dotenv()

from vitssm.data import get_dataloaders_next_frame
from vitssm.models import build_model
from vitssm.engine.tasks import NextFrameUPTEngine


wandb.login()
wandb.require("core")


def main(config: str):   
    config_path = Path(os.environ["CONFIG_DIR"]) / config
    
    dit_config = OmegaConf.load(config_path)
    
    with wandb.init(
        job_type="train",
        entity=os.environ["WANDB_ENTITY"],
        project=dit_config.project,
        group=dit_config.group,
        name=dit_config.name,
        id=dit_config.name + "_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
        tags=dit_config.tags,
        config=OmegaConf.to_container(dit_config, resolve=True),
        resume="allow",
    ):      
        #print(wandb.config)
        run_config = OmegaConf.create(dict(wandb.config))
        model = build_model(run_config.model)
        engine = NextFrameUPTEngine(model=model, run_object=run_config)

        #if wandb.run.resumed:
        #    engine._resume_checkpoint()

        train_loader, val_loader = get_dataloaders_next_frame(run_config.dataset)

        engine.train(train_loader, val_loader)


if __name__ == '__main__':
    
    args = argparse.ArgumentParser()
    
    args.add_argument("--config", type=str, default="dit_config.yml")
    
    args = args.parse_args()
    
    main(
        config=args.config,
    )
