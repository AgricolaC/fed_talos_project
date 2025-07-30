import torch
from pathlib import Path
from project_utils.data_split  import load_cifar100, iid_partition, noniid_partition
from federated_utils.client   import Client
from federated_utils.server   import Server
from project_utils.train_utils import seed_all
from project_utils.model_registry import MODEL_REGISTRY
from torch.utils.data import Subset

def train_federated(config):
    seed_all(config.get("seed",42))
    # device + model constructor
    
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Using device: {device}")
    
    model_cfg = config.get("model")
    
    if isinstance(model_cfg, str):
        if model_cfg not in MODEL_REGISTRY:
            raise KeyError(f"[train_federated] Unknown model string '{model_cfg}'. "
                       f"Available: {list(MODEL_REGISTRY.keys())}")
        config["model"] = MODEL_REGISTRY[model_cfg]

    if not callable(config["model"]):
        raise KeyError("[train_federated] config['model'] must be a callable class")
    
    config["model_constructor"] = lambda: config["model"](
        num_classes     = config["num_classes"],
        frozen_backbone = config.get("frozen_backbone", False)
    )
    
    print(f"[Federated] Using model class: {config['model'].__name__}")

    print("[Federated] Loading data...")
    train_set, val_set, test_set = load_cifar100(config["data_root"], config["val_split"])
    split = config.get("data_split","iid").lower()
    if split == "iid":
        client_indices = iid_partition(train_set, config["num_clients"])
    elif split == "noniid":
        client_indices = noniid_partition(train_set,
                                           config["num_clients"],
                                           config["classes_per_client"])
    else:
        raise ValueError(f"[Federated] Unknown data_split: {split}")

    # build clients
    clients = [
        Client(
            Subset(train_set, idx_list),   
            config,
            cid=cid
        )   
        for cid, idx_list in client_indices.items()
    ]
    
    for cid, ds in enumerate(clients):
        assert isinstance(ds.dataset, torch.utils.data.Dataset)

    val_loader  = torch.utils.data.DataLoader(val_set,  batch_size=config["batch_size"], shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=config["batch_size"], shuffle=False)

    global_model = config["model_constructor"]().to(device)
    print("[Federated] Global model initialized")

    # server orchestration
    server = Server(global_model, clients, val_loader, test_loader, config)
    return server.train()
