import wandb
from vitssm.data.datasets import VideoMDSpritesDataset


wandb.login()

wandb.init(
    project="vitssm",
    group="test",
    name="test",
    id="test",
    resume="allow",
)
wandb.log({"test": 1})
wandb.log({"test": 2})
wandb.log({"test": 3})

dataset = VideoMDSpritesDataset("data/VMDsprites", download=True, train=True, fold=0)

wandb.log({"len": len(dataset)})
wandb.log({"sample": dataset[0]})
wandb.log({"sample": dataset[1]})

wandb.finish()