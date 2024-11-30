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

class Optimized_ModelTrainer:
    def __init__(self, device: torch.device):
        self.device = device
        self.loss_func = nn.CrossEntropyLoss()
        self.model_states = {}  

    def get_config_key(self, config: Dict) -> str:

        return str(sorted(config.items()))

    def train_model(self, model: nn.Module, config: Dict, resource: float,
                   train_loader: torch.utils.data.DataLoader,
                   total_trained_time: float = 0) -> float:

        config_key = self.get_config_key(config)

        if config_key in self.model_states:
            model.load_state_dict(self.model_states[config_key]['state_dict'])
            total_trained_time = self.model_states[config_key]['trained_time']

        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        start_time = time.time()

        while time.time() - start_time < resource:
            for images, labels in train_loader:
                current_time = time.time() - start_time
                if current_time >= resource:
                    break

                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = model.feed_forward(images)
                loss = self.loss_func(outputs, labels)
                loss.backward()
                optimizer.step()

        '''save the models current state'''
        total_trained_time += (time.time() - start_time)
        self.model_states[config_key] = {
            'state_dict': model.state_dict(),
            'trained_time': total_trained_time
        }

        return total_trained_time

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
