import torch 
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Subset, random_split
from collections import defaultdict
import random

def load_cifar100(data_root='./data', val_split=0.1):
    # Normalization values for CIFAR-100
    mean = (0.507, 0.487, 0.441)
    std = (0.267, 0.256, 0.276)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(112, scale=(0.6, 1.0)),  # tighter scale to preserve spatial integrity
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(112),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    full_train = datasets.CIFAR100(root=data_root, train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR100(root=data_root, train=False, download=True, transform=test_transform)

    val_len = int(len(full_train) * val_split)
    train_len = len(full_train) - val_len
    train_set, val_set = random_split(full_train, [train_len, val_len])

    # Apply test_transform to val set (not augmented)
    val_set.dataset.transform = test_transform

    return train_set, val_set, test_set

def iid_partition(dataset, num_clients):
    num_items = len(dataset) // num_clients
    all_indices = list(range(len(dataset)))
    random.shuffle(all_indices)
    
    client_dict = {i: all_indices[i * num_items:(i + 1) * num_items] for i in range(num_clients)}
    return client_dict

def noniid_partition(dataset, num_clients, nc_per_client):
    labels = np.array(dataset.dataset.targets)[dataset.indices] if isinstance(dataset, Subset) else np.array(dataset.targets)
    data_by_class = defaultdict(list)

    for idx, label in enumerate(labels):
        data_by_class[label].append(idx)

    for cls in data_by_class:
        random.shuffle(data_by_class[cls])

    client_dict = {i: [] for i in range(num_clients)}
    class_list = list(data_by_class.keys())

    # Assign classes randomly to clients
    for client_id in range(num_clients):
        chosen_classes = random.sample(class_list, nc_per_client)
        for cls in chosen_classes:
            num_available = len(data_by_class[cls])
            take = min(num_available, len(labels) // num_clients // nc_per_client)
            client_dict[client_id].extend(data_by_class[cls][:take])
            data_by_class[cls] = data_by_class[cls][take:]

    return client_dict
