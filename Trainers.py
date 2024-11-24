import time
import torch
import torch.nn as nn
from typing import Dict


class ModelTrainer:
    def __init__(self, device : torch.device):
        self.device = device
        self.loss_func = nn.CrossEntropyLoss()

    def train_model(self, model: nn.Module, config: Dict, resource: float,
                   train_loader: torch.utils.data.DataLoader) -> None:
        """Train model with early stopping based on time limit"""
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        start_time = time.time()

        while time.time() - start_time < resource:
            for images, labels in train_loader:
                if time.time() - start_time >= resource:
                    break

                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = model.feed_forward(images)
                loss = self.loss_func(outputs, labels)
                loss.backward()
                optimizer.step()

    @torch.no_grad()
    def evaluate_model(self, model: nn.Module,
                      test_loader: torch.utils.data.DataLoader) -> float:
        """Evaluate model on test set"""
        model.eval()
        total_loss = 0
        total_batches = 0

        for images, labels in test_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = model.feed_forward(images)
            loss = self.loss_func(outputs, labels)
            total_loss += loss.item()
            total_batches += 1

        return total_loss / total_batches