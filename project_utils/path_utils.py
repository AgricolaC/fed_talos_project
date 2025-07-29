from pathlib import Path
import yaml

def get_project_root(anchor_filename="config.yaml") -> Path:
    """Traverse upward until it finds the anchor file."""
    current = Path.cwd()
    for _ in range(10):  # Limit traversal depth
        if (current / anchor_filename).exists():
            return current
        current = current.parent
    raise RuntimeError(f"Could not find {anchor_filename} in parent directories.")

def load_config_with_absolute_paths(anchor_file="configs/config.yaml") -> dict:
    """Loads config and resolves relative paths against project root."""
    anchor_path = Path(anchor_file).resolve()

    # Assume config lives inside /configs, project root is one level up
    project_root = anchor_path.parent.parent

    with open(anchor_path, "r") as f:
        config = yaml.safe_load(f)

    for key in ["data_root", "log_dir", "wandb_dir", "checkpoint_path"]:
        if key in config:
            config[key] = str((project_root / Path(config[key])).resolve())

    return config