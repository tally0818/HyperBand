import numpy as np
from typing import Dict, Tuple
import torch
import torch.nn as nn
from Trainers import ModelTrainer

from search_space import SearchSpace

class RandomSearch:
  def __init__(self, R : int, eta : float):
    self.R = R # Same R from HyperBand -> total R*(s_max+1)^2 Resources are allocated
    self.eta = eta # same
    self.s_max = int(np.floor(np.log(R) / np.log(eta)))
    self.total_resource = self.R * (self.s_max + 1) ** 2
    self.avg_resource = self.total_resource / (sum([int(np.ceil((self.s_max + 1) * self.eta ** s / (s + 1))) for s in range(self.s_max+1, -1, -1)]))
    self.n = self.total_resource // self.avg_resource # numbers of configs randomsearch will consider
    self.device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.trainer = ModelTrainer(self.device)

  def run_then_return_val_loss(self, model_class: type[nn.Module],
                               config: Dict, resource: float,
                               train_loader: torch.utils.data.DataLoader,
                               test_loader: torch.utils.data.DataLoader) -> float:
    """Run a single trial with given configuration and resource allocation"""
    model = model_class(config).to(self.device)
    self.trainer.train_model(model, config, resource, train_loader)
    return self.trainer.evaluate_model(model, test_loader)

  def optimize(self, model_class: type[nn.Module],
               search_space: SearchSpace,
               train_loader: torch.utils.data.DataLoader,
               test_loader: torch.utils.data.DataLoader) -> Tuple[Dict, float]:
    best_config = {}
    best_loss = np.inf
    configs = search_space.sample(int(self.n))
    for config in configs:
      loss = self.run_then_return_val_loss(model_class, config, self.avg_resource, train_loader, test_loader)
      if loss < best_loss:
        best_loss = loss
        best_config = config
        print(f'current minimum loss: {best_loss:.4f}')
    return best_config, best_loss
