import torch
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, LinearLR
from models.dino_vits16 import DINO_ViT
from project_utils.data_split import load_cifar100
from project_utils.train_utils import train_one_epoch, evaluate
from project_utils.wandb_logger import log_metrics
import os

def get_scheduler(optimizer, scheduler_type, config):
    if scheduler_type == "cosine":
        return CosineAnnealingLR(optimizer, T_max=config["epochs"])
    elif scheduler_type == "step":
        return StepLR(optimizer, step_size=10, gamma=0.5)
    elif scheduler_type == "linear":
        return LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=config["epochs"])
    elif scheduler_type == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    print(f"Using scheduler: {scheduler_type} with parameters: {config.get(scheduler_type, {})}")


def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and torch.backends.mps.is_available():
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    print(f"Using device: {device}")
    print(f"Data root: {config['data_root']} | Logs: {config['log_dir']} | Checkpoint: {config['checkpoint_path']}")

    train_set, val_set, test_set = load_cifar100(data_root= config["data_root"], val_split=config["val_split"])
    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config["batch_size"])
    test_loader = DataLoader(test_set, batch_size=config["batch_size"])

    model = DINO_ViT(num_classes=config["num_classes"], frozen_backbone=config.get("frozen_backbone", False)).to(device)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters.")
    print(f"Frozen backbone: {config.get('frozen_backbone', False)}")
    
    if config.get("use_checkpoint", False) and os.path.exists(config["checkpoint_path"]):
        print(f"Resuming from checkpoint: {config['checkpoint_path']}")
        model.load_state_dict(torch.load(config["checkpoint_path"]))
    
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["lr"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"]
    )
    scheduler = get_scheduler(optimizer, config["scheduler"], config)
    print(f"Using scheduler: {config['scheduler']} with initial learning rate: {config['lr']}")
    
    best_val_acc = 0.0
    epoch_times, val_accs = [], []

    for epoch in range(config["epochs"]):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        val_accs.append(val_acc)

        epoch_duration = time.time() - start_time
        epoch_times.append(epoch_duration)
        eta = (sum(epoch_times) / len(epoch_times)) * (config["epochs"] - epoch - 1)

        log_metrics(epoch, train_loss, train_acc, val_loss, val_acc, scheduler.get_last_lr()[0], epoch_duration, eta, config)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"Best model updated at epoch {epoch + 1} with Val Acc = {val_acc:.4f}")
            # Ensure checkpoint directory exists
            os.makedirs(os.path.dirname(config["checkpoint_path"]), exist_ok=True)
            torch.save(model.state_dict(), config["checkpoint_path"])

        if (epoch + 1) % 5 == 0:
            print(f"Checkpoint saved at epoch {epoch + 1}")

        scheduler.step()

    # Final test evaluation
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"\n TRAINING COMPLETED | Final Test Accuracy: {test_acc:.4f} | Total Time: {sum(epoch_times)/60:.2f} min")

    if not config["use_wandb"]:
        plt.plot(val_accs)
        plt.title("Validation Accuracy over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Val Accuracy")
        plt.grid(True)
        plt.savefig("val_accuracy.png")
        plt.show()

    return test_acc
