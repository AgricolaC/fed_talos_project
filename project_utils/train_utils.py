# train_utils.py
import random
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, LinearLR, SequentialLR
import os
import numpy as np


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    
def save_checkpoint(model, optimizer, scheduler, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, path)
    
def get_scheduler(optimizer, scheduler_type, config):
    if scheduler_type == "cosine":
        warmup_epochs = config.get("warmup_epochs", 0)
        if warmup_epochs > 0:
            main_sched = CosineAnnealingLR(optimizer, T_max=config["epochs"] - warmup_epochs)
            warmup_sched = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
            return SequentialLR(optimizer, schedulers=[warmup_sched, main_sched], milestones=[warmup_epochs])
        else:
            return CosineAnnealingLR(optimizer, T_max=config["epochs"])
    elif scheduler_type == "step":
        return StepLR(optimizer, step_size=10, gamma=0.5)
    elif scheduler_type == "linear":
        return LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=config["epochs"])
    elif scheduler_type == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, correct = 0, 0
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = F.cross_entropy(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += outputs.argmax(1).eq(y).sum().item()

    avg_loss = total_loss / len(loader.dataset)
    avg_acc = correct / len(loader.dataset)
    return avg_loss, avg_acc


def evaluate(model, loader, device):
    model.eval()
    total_loss, correct = 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = F.cross_entropy(outputs, y)
            total_loss += loss.item() * x.size(0)
            correct += outputs.argmax(1).eq(y).sum().item()

    avg_loss = total_loss / len(loader.dataset)
    avg_acc = correct / len(loader.dataset)
    return avg_loss, avg_acc
