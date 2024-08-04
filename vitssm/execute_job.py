import argparse
import json
from omegaconf import OmegaConf, DictConfig
from pathlib import Path
from typing import Union
from vitssm.models import build_model
from vitssm.engine.tasks import ActionRecognitionEngine
from vitssm.data import build_dataloaders


def load_config(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


def get_configs(
    base_config_path: Union[str, Path],
    run_config_path: Union[str, Path],
) -> DictConfig:

    base_config = OmegaConf.load(base_config_path)
    run_config = OmegaConf.load(run_config_path)
    config = OmegaConf.merge(base_config, run_config)

    return config


def launch_job(config):
    model = build_model(config)
    train_loader, eval_loader = build_dataloaders(config)
    engine = ActionRecognitionEngine(model, config)
    engine.train(train_loader, eval_loader)
    engine.run.finish()


def main():
    parser = argparse.ArgumentParser(description='Launch a job with merged configurations.')
    parser.add_argument('bc', type=str, help='Path to the base configuration file')
    parser.add_argument('rc', type=str, help='Path to the run configuration file')
    args = parser.parse_args()

    config = get_configs(args.bc, args.rc)
    launch_job(config)


if __name__ == '__main__':
    main()