import wandb

def init_wandb(config):
    run = wandb.init(
        project=config["project_name"],
        entity=config["entity"],
        config=config,
        name=f"{config['model']}_lr{config['lr']}_bs{config['batch_size']}",
        dir=config.get("wandb_dir", "./logs/wandb"),
    )
    return run.config

def log_metrics(epoch, train_loss, train_acc, val_loss, val_acc, lr, epoch_time, eta, config):
    if config["use_wandb"]:
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": lr,
            "epoch_time_sec": epoch_time,
            "eta_min": eta / 60
        })
    else:
        print(
            f"[Epoch {epoch + 1}/{config['epochs']}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"Time: {epoch_time:.2f}s | ETA: {eta / 60:.2f} min"
        )

