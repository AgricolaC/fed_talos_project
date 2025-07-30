import torch
from pathlib import Path
from project_utils.data_split import load_cifar100, iid_partition, noniid_partition
from federated_utils.client import Client
from federated_utils.server import Server
from project_utils.train_utils import seed_all

def train_federated(config):
    seed_all(config.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config["device"] = device
    config["model_constructor"] = lambda: config["model"](
        num_classes=config["num_classes"],
        frozen_backbone=config.get("frozen_backbone", False)
    )

    # load data + split into clients
    train_set, val_set, test_set = load_cifar100(config["data_root"], config["val_split"])
    if config["data_split"] == "iid":
        client_datasets = iid_partition(train_set, config["num_clients"])
    else:
        client_datasets = noniid_partition(train_set, config["num_clients"], config["classes_per_client"])

    clients = [
        Client(cid, ds, config)
        for cid, ds in enumerate(client_datasets)
    ]

    val_loader  = torch.utils.data.DataLoader(val_set,  batch_size=config["batch_size"])
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=config["batch_size"])

    # init global model
    global_model = config["model_constructor"]().to(device)

    server = Server(global_model, clients, val_loader, test_loader, config)
    return server.train()
