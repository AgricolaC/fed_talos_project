# train_utils.py

import torch
import torch.nn.functional as F

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
