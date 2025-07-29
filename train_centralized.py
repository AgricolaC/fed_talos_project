# train_centralized.py

import torch
import time
import wandb
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from models.dino_vits16 import DINO_ViT
from project_utils.data_split import load_cifar100
from project_utils.train_utils import train_one_epoch, evaluate
from project_utils.wandb_logger import init_wandb, log_metrics
from config import config


def get_scheduler(optimizer, scheduler_type, config):
    if scheduler_type == "cosine":
        return CosineAnnealingLR(optimizer, T_max=config["epochs"])
    elif scheduler_type == "step":
        return StepLR(optimizer, step_size=10, gamma=0.5)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def train():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    train_set, val_set, test_set = load_cifar100()
    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config["batch_size"])
    test_loader = DataLoader(test_set, batch_size=config["batch_size"])

    model = DINO_ViT(num_classes=100, frozen_backbone=config["frozen_backbone"]).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["lr"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"]
    )
    scheduler = get_scheduler(optimizer, config["scheduler"], config)

    if config["use_wandb"]:
        init_wandb(config)

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
            print(f"📌 Best model updated at epoch {epoch + 1} with Val Acc = {val_acc:.4f}")

        if (epoch + 1) % 5 == 0:
            print(f"💾 Checkpoint saved at epoch {epoch + 1}")

        scheduler.step()

    # Final test evaluation
    test_loss, test_acc = evaluate(model, test_loader, device)
    if config["use_wandb"]:
        wandb.log({"test_loss": test_loss, "test_acc": test_acc})
        wandb.finish()

    print(f"\n✅ TRAINING COMPLETED | Final Test Accuracy: {test_acc:.4f} | Total Time: {sum(epoch_times)/60:.2f} min")

    if not config["use_wandb"]:
        plt.plot(val_accs)
        plt.title("Validation Accuracy over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Val Accuracy")
        plt.grid(True)
        plt.savefig("val_accuracy.png")
        plt.show()


if __name__ == "__main__":
    train()
