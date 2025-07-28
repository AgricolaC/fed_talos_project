import torch 
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Subset, random_split
from collections import defaultdict
import random

def load_cifar100(data_root= './data', val_split = 0.1):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),  # DINO ViT expects 224x224 input
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))
    ])
    
    full_train = datasets.CIFAR100(root=data_root, train= True, download= True, transform= train_transform)
    test_set = datasets.CIFAR100(root=data_root, train= False, download= True, transform= test_transform)
    
    val_len = int(len(full_train) * val_split)
    train_len = len(full_train) - val_len   
    train_set, val_set = random_split(full_train, [train_len, val_len])
    
    val_set.dataset.transform = test_transform
    
    return train_set, val_set, test_set

def iid_partition(dataset, num_clients):
    num_items = len(dataset) // num_clients
    all_indices = list(range(len(dataset)))
    random.shuffle(all_indices)
    
    client_dict = {i: all_indices[i * num_items:(i + 1) * num_items] for i in range(num_clients)}
    return client_dict

def noniid_partition(dataset, num_clients, nc_per_client):
    labels = np.array(dataset.dataset.targets)[dataset.indices] if isinstance(dataset, Subset) else dataset.targets
    data_by_class = defaultdict(list)
    
    for idx, label in enumerate(labels):
        data_by_class[label].append(idx)
    
    # Shuffle within each class
    for label in data_by_class:
        random.shuffle(data_by_class[label])
    
    class_ids = list(data_by_class.keys())
    random.shuffle(class_ids)

    client_dict = {i: [] for i in range(num_clients)}
    class_splits = np.array_split(class_ids, num_clients)

    for client_id, class_subset in enumerate(class_splits):
        for cls in class_subset[:nc_per_client]:
            num_samples = len(data_by_class[cls]) // (num_clients // len(class_subset))
            client_dict[client_id].extend(data_by_class[cls][:num_samples])
            data_by_class[cls] = data_by_class[cls][num_samples:]

    return client_dict
