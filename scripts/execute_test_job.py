import wandb


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

wandb.finish()