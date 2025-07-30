import copy
import torch
from torch.utils.data import DataLoader
from project_utils.train_utils import train_one_epoch, evaluate, get_scheduler, seed_all

class Client:
    def __init__(self, cid, dataset, config):
        self.cid = cid
        self.dataset = dataset
        self.config = config

        # each client gets its own loader
        self.loader = DataLoader(
            self.dataset,
            batch_size=config["batch_size"],
            shuffle=True,
        )

    def local_train(self, global_model_state):
        """
        1) instantiate a fresh model and load global weights
        2) run J local epochs
        3) return the delta (new_state - global_state), and a drift metric
        """
        # reproducibility per client
        seed_all(self.config.get("seed", 42) + self.cid)

        # 1) model, optimizer, scheduler
        model = self.config["model_constructor"]().to(self.config["device"])
        model.load_state_dict(global_model_state, strict=True)

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.config["lr"],
            momentum=self.config["momentum"],
            weight_decay=self.config["weight_decay"],
        )
        scheduler = get_scheduler(optimizer, self.config["scheduler"], self.config)

        # 2) local epochs
        for _ in range(self.config["local_steps"]):
            train_one_epoch(model, self.loader, optimizer, self.config["device"])
            if scheduler: scheduler.step()

        # 3) compute weight delta and drift
        new_state = model.state_dict()
        delta = {k: new_state[k] - global_model_state[k] for k in new_state}

        # L2 drift
        drift = torch.sqrt(
            sum((delta[k].float()**2).sum() for k in delta)
        ).item()

        return delta, drift
