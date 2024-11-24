import numpy as np
from typing import Dict, List, Tuple
import torch
from Trainers import ModelTrainer
from Models import BaseModel
from search_space import SearchSpace



def top_k(configs: List[Dict], losses: List[float], k: int) -> Tuple[List[Dict], List[float]]:
    """Return top k configurations based on losses"""
    sorted_pairs = sorted(zip(configs, losses), key=lambda x: x[1])
    configs, losses = zip(*sorted_pairs[:k])
    return list(configs), list(losses)

class HyperBand:
    def __init__(self, R : int, eta : float):
        self.R = R
        self.eta = eta
        self.device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.trainer = ModelTrainer(self.device)

    def run_then_return_val_loss(self, config: Dict, resource: float,
                 train_loader: torch.utils.data.DataLoader,
                 test_loader: torch.utils.data.DataLoader) -> float:
        """Run a single trial with given configuration and resource allocation"""
        model = BaseModel(config).to(self.device)
        self.trainer.train_model(model, config, resource, train_loader)
        return self.trainer.evaluate_model(model, test_loader)

    def optimize(self, search_space: SearchSpace,
                train_loader: torch.utils.data.DataLoader,
                test_loader: torch.utils.data.DataLoader) -> Tuple[Dict, float]:
        """Main HyperBand optimization loop"""
        s_max = int(np.floor(np.log(self.R) / np.log(self.eta)))
        B = (s_max + 1) * self.R

        best_configs = []
        best_losses = []

        for s in range(s_max, -1, -1):
            n = int(np.ceil(B * (self.eta ** s) / (self.R * (s + 1))))
            r = self.R * (self.eta ** -s)

            # Successive Halving with (n,r)
            configs = search_space.sample(n)

            for i in range(s + 1):
                n_i = int(np.floor(n * (self.eta ** -i)))
                r_i = r * (self.eta ** i)

                losses = [self.run_then_return_val_loss(config, r_i, train_loader, test_loader)
                         for config in configs]

                k = max(1, int(np.floor(n_i / self.eta)))
                configs, losses = top_k(configs, losses, k)

            best_idx = np.argmin(losses)
            best_configs.append(configs[best_idx])
            best_losses.append(losses[best_idx])
            print(f'Observed minimum loss: {losses[best_idx]:.4f}')

        final_best_idx = np.argmin(best_losses)
        return best_configs[final_best_idx], best_losses[final_best_idx]