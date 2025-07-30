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
        self.best_val_acc = 0.0

        print(f"[Server] {len(clients)} clients registered")
        print(f"[Server] Validation size={len(val_loader.dataset)}, Test size={len(test_loader.dataset)}")

    def aggregate(self, deltas):
        state = self.global_model.state_dict()
        avg_delta = {}
        K = len(deltas)
        print(f"[Server] Aggregating updates from {K} clients")
        for k in state:
            stacked = torch.stack([d[k].to(self.device) for d in deltas], dim=0)
            avg_delta[k] = stacked.mean(dim=0)
        for k in state:
            state[k] = state[k].to(self.device) + avg_delta[k]
        self.global_model.load_state_dict(state)

    def evaluate_global(self, loader):
        loss, acc = evaluate(self.global_model, loader, self.device)
        return loss, acc

    def train(self):
        rounds = self.config.get("rounds", 1)
        frac   = self.config.get("client_fraction", 1.0)
        print(f"[Server] Starting federated training for {rounds} rounds (C={frac:.2f})")
        for rnd in range(1, rounds+1):
            m = max(1, int(frac * len(self.clients)))
            selected = torch.randperm(len(self.clients))[:m].tolist()
            print(f"\n[Round {rnd}] Selected clients: {selected}")

            deltas, drifts = [], []
            base_state = self.global_model.state_dict()
            for idx in selected:
                d, drift = self.clients[idx].local_train(base_state)
                deltas.append(d); drifts.append(drift)

            # aggregate & drift
            self.aggregate(deltas)
            avg_drift = sum(drifts) / len(drifts)
            print(f"[Round {rnd}] → Avg client L2‐drift: {avg_drift:.4f}")

            # validate
            val_loss, val_acc = self.evaluate_global(self.val_loader)
            print(f"[Round {rnd}] Validation Acc = {val_acc:.4f} (loss={val_loss:.4f})")
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                print(f"[Server] ✔ New best global model @round {rnd}: val_acc={val_acc:.4f}")
                save_checkpoint(
                    self.global_model,
                    optimizer=None,
                    scheduler=None,
                    path=self.config["checkpoint_path"]
                )

        # final test
        test_loss, test_acc = self.evaluate_global(self.test_loader)
        print(f"\n[Server] FINAL GLOBAL TEST  Acc={test_acc:.4f}  Loss={test_loss:.4f}")
        return test_acc
