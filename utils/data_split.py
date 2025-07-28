import torch 
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Subset, random_split
from collections import defaultdict
import random

def load_cifar100(data_root= './data', val_split = 0.1):
    transform = transforms.compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    full_train = datasets.CIFAR100(root=data_root, train= True, download= True, transform= transform)
    test_set = datasets.CIFAR100(root=data_root, train= False, download= True, transform= transform)
    
    val_len = int(len(full_train) * val_split)
    train_len = len(full_train) - val_len   
    train_set, val_set = random_split(full_train, [train_len, val_len])
    
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
