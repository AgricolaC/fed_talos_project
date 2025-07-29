import torch
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from models.dino_vits16 import DINO_ViT
from project_utils.data_split import load_cifar100
from project_utils.train_utils import train_one_epoch, evaluate, get_scheduler
from project_utils.wandb_logger import log_metrics
import os
from pathlib import Path
from project_utils.train_utils import save_checkpoint, seed_all
def train(config):
    seed_all(config.get("seed", 42))
    # Device setup
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Using device: {device}")
    
    # Resolve paths
    checkpoint_path = Path(config["checkpoint_path"]).resolve()
    log_dir = Path(config["log_dir"]).resolve()
    wandb_dir = Path(config["wandb_dir"]).resolve()
    data_root = Path(config["data_root"]).resolve()

    print(f"Data: {data_root} | Logs: {log_dir} | Checkpoint: {checkpoint_path}")
    
    # Load and process data
    train_set, val_set, test_set = load_cifar100(data_root= config["data_root"], val_split=config["val_split"])
    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config["batch_size"])
    test_loader = DataLoader(test_set, batch_size=config["batch_size"])

    model = DINO_ViT(num_classes=config["num_classes"], frozen_backbone=config.get("frozen_backbone", False)).to(device)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters.")
    print(f"Frozen backbone: {config.get('frozen_backbone', False)}")
    
    # Checkpoint loading
    if config.get("use_checkpoint", False) and checkpoint_path.exists():
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    # Optimizer & Scheduler
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["lr"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"]
    )
    scheduler = get_scheduler(optimizer, config["scheduler"], config)
    print(f"Using scheduler: {config['scheduler']} with initial learning rate: {config['lr']}")
    
    best_val_acc = 0.0
    epoch_times, val_accs, val_losses = [], [], []

    # Training loop
    for epoch in range(config["epochs"]):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        val_accs.append(val_acc)
        val_losses.append(val_loss)

        epoch_duration = time.time() - start_time
        epoch_times.append(epoch_duration)
        eta = (sum(epoch_times) / len(epoch_times)) * (config["epochs"] - epoch - 1)

        log_metrics(epoch, train_loss, train_acc, val_loss, val_acc, scheduler.get_last_lr()[0], epoch_duration, eta, config)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"Best model updated at epoch {epoch + 1} with Val Acc = {val_acc:.4f}")
            model_tag = log_dir / f"model_epoch{epoch + 1}_valacc{val_acc:.4f}.pth"
            model_tag.parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(model, optimizer, scheduler, model_tag)
            save_checkpoint(model, optimizer, scheduler, checkpoint_path)

        if (epoch + 1) % 5 == 0:
            print(f"Checkpoint saved at epoch {epoch + 1}")

        scheduler.step()

    # Final test evaluation
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"\n TRAINING COMPLETED | Final Test Accuracy: {test_acc:.4f} | Total Time: {sum(epoch_times)/60:.2f} min")

    # Plotting results
    if not config["use_wandb"]:
        plt.figure(figsize=(12, 6))
        plt.plot(val_loss)
        plt.title("Validation Loss over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Val Loss")
        plt.grid(True)
        plt.savefig("val_loss.png")
        plt.show()
    if not config["use_wandb"]:
        plt.figure(figsize=(12, 6))
        plt.plot(val_accs)
        plt.title("Validation Accuracy over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Val Accuracy")
        plt.grid(True)
        plt.savefig("val_accuracy.png")
        plt.show()

    return test_acc
