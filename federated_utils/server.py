import copy
import torch
from project_utils.train_utils import save_checkpoint, evaluate

class Server:
    def __init__(self, model, clients, val_loader, test_loader, config):
        self.global_model = model
        self.clients      = clients
        self.val_loader   = val_loader
        self.test_loader  = test_loader
        self.config       = config
        self.device       = config["device"]

        # keep track
        self.best_val_acc = 0.0

    def aggregate(self, deltas):
        """
        Simple FedAvg: average each tensor delta[k] across clients,
        then add to global weights.
        """
        global_state = self.global_model.state_dict()
        avg_delta = {}
        K = len(deltas)
        for k in global_state:
            stacked = torch.stack([d[k] for d in deltas], dim=0)
            avg_delta[k] = stacked.mean(dim=0)

        # apply
        for k in global_state:
            global_state[k] = global_state[k] + avg_delta[k]
        self.global_model.load_state_dict(global_state)

    def evaluate_global(self, loader):
        return evaluate(self.global_model, loader, self.device)

    def train(self):
        for rnd in range(self.config["rounds"]):
            # 1) select a fraction C of clients
            m = max(1, int(self.config["client_fraction"] * len(self.clients)))
            selected = torch.randperm(len(self.clients))[:m].tolist()

            deltas, drifts = [], []
            for idx in selected:
                delta, drift = self.clients[idx].local_train(self.global_model.state_dict())
                deltas.append(delta)
                drifts.append(drift)

            # 2) aggregate
            self.aggregate(deltas)

            # 3) log client drift
            avg_drift = sum(drifts) / len(drifts)
            print(f"[Round {rnd+1}] ⟳ avg client L₂‐drift: {avg_drift:.4f}")

            # 4) validate global model
            val_loss, val_acc = self.evaluate_global(self.val_loader)
            print(f"[Round {rnd+1}] Validation Acc: {val_acc:.4f}")

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                print(f"→ New best global model at round {rnd+1}, val acc={val_acc:.4f}")
                save_checkpoint(
                    self.global_model,
                    optimizer=None,
                    scheduler=None,
                    path=self.config["checkpoint_path"]
                )

        # final test
        test_loss, test_acc = self.evaluate_global(self.test_loader)
        print(f"\n FINAL GLOBAL TEST ACC: {test_acc:.4f}")
        return test_acc
