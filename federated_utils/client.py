import copy
import torch
from torch.utils.data import DataLoader, Dataset
from project_utils.train_utils import train_one_epoch, get_scheduler, seed_all

class Client:
    def __init__(self, dataset, config, cid=None):
        # sanity checks
        if not isinstance(dataset, Dataset):
            raise TypeError(f"[Client {cid}] Expected a torch.utils.data.Dataset, got {type(dataset)}")
        required = ["batch_size", "model_constructor", "lr", "momentum", "weight_decay", "scheduler", "local_steps", "device"]
        missing = [k for k in required if k not in config]
        if missing:
            raise KeyError(f"[Client {cid}] Missing config keys: {missing}")
        if not callable(config["model_constructor"]):
            raise TypeError(f"[Client {cid}] 'model_constructor' must be callable()")

        self.cid = cid
        self.dataset = dataset
        self.config = config
        print(f"[Client {self.cid}] Initialized with {len(dataset)} samples")

        # DataLoader
        self.loader = DataLoader(
            self.dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config.get("num_workers", 0),
            pin_memory=(config.get("device").type=="cuda")
        )

    def local_train(self, global_model_state):
        print(f"[Client {self.cid}] Starting local training")
        #  reproducibility
        seed_all(self.config.get("seed", 42) + (self.cid or 0))

        # 1) instantiate model
        model = self.config["model_constructor"]().to(self.config["device"])
        model.load_state_dict(global_model_state, strict=True)

        # optimizer + scheduler
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.config["lr"],
            momentum=self.config["momentum"],
            weight_decay=self.config["weight_decay"]
        )
        scheduler = get_scheduler(optimizer, self.config["scheduler"], cfg={ "warmup_steps": 2, "total_steps": self.config["local_steps"] })

        # 2) local epochs
        for step in range(self.config["local_steps"]):
            loss, acc = train_one_epoch(model, self.loader, optimizer, self.config["device"])
            print(f"[Client {self.cid}]  Step {step+1}/{self.config['local_steps']}  loss={loss:.4f} acc={acc:.4f}")
            if scheduler:
                scheduler.step()

        # 3) compute delta + drift
        new_state = model.state_dict()
        delta = {}
        for k in new_state:
            delta[k] = new_state[k].cpu() - global_model_state[k].cpu()

        # scalar L2 drift
        drift = torch.sqrt(sum((d.float()**2).sum() for d in delta.values())).item()
        print(f"[Client {self.cid}] Finished locals — L2 drift={drift:.4f}")
        return delta, drift

