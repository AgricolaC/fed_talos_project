import wandb
import yaml

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
        if config is None:
            raise ValueError(f"YAML config at '{path}' is empty or invalid.")
        return config

def init_wandb(config):
    wandb.init(
        project=config["project_name"],
        entity=config["entity"],
        config=config,
        name=f"{config['model']}_lr{config['lr']}_bs{config['batch_size']}"
    )
    return wandb.config
