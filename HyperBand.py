import time
import numpy as np
from typing import Dict, Any, Union, List
from search_space import SearchSpace
from Models import BaseModel
import torch
import torch.nn as nn


def top_k(configs, losses, k):
  '''
  a function that takes a set of configurations as well as their associated losses and returns thr top k performing configurations with their losses
  '''
  sorted_configs_losses = sorted(zip(configs, losses), key=lambda x: x[1])
  return [config for config, loss in sorted_configs_losses[:k]], [loss for config, loss in sorted_configs_losses[:k]]

def run_then_return_val_loss(t: dict, r, train_loader, test_loader):
  '''
  a function that takes a hyperparameter configuration t and resource allocation r as input and
  returns the validation loss after training the configuration for the allocated resources.
  '''

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model = BaseModel(t).to(device)
  batch_size = 50
  learning_rate = t['lr']
  loss_func = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

  start_time = time.time()
  time_limit = r

  epoch = 0
  while time.time() - start_time < time_limit:
    epoch += 1
    for j, [image, label] in enumerate(train_loader):
      x = image.to(device)
      y = label.to(device)

      optimizer.zero_grad()
      output = model.feed_forward(x)
      loss = loss_func(output, y)
      loss.backward()
      optimizer.step()

      if time.time() - start_time >= time_limit:
        break
    if time.time() - start_time >= time_limit:
      break

  total_loss = 0
  num_batches = 0
  model.eval()
  with torch.no_grad():
    for image, label in test_loader:
      x = image.to(device)
      y = label.to(device)
      output = model.feed_forward(x)
      loss = loss_func(output, y)
      total_loss += loss.item()
      num_batches += 1

  return total_loss / num_batches

def HyperBand(search_space : SearchSpace, R, eta, train_loader, test_loader):
  s_max = int(np.floor(np.log(R) / np.log(eta)))
  B = (s_max + 1) * R
  opt_arms = []
  opt_losses = []
  for s in range(s_max, -1, -1):
    n = int(np.ceil(B * (eta ** s) / (R * (s + 1))))
    r = R * (eta ** -s)
    # begin SH with (n,r) inner loop
    T = search_space.sample(n)
    for i in range(s + 1):
      n_i = int(np.floor(n * (eta ** -i)))
      r_i = r * (eta ** i)
      observed_losses = []
      for config in T:
        loss = run_then_return_val_loss(config, r_i, train_loader, test_loader)
        observed_losses.append(loss)
      T, observed_losses = top_k(T, observed_losses, max(1,int(np.floor(n_i / eta))))
    observed_losses = np.array(observed_losses)
    opt_arm = T[np.argmin(observed_losses)]
    opt_loss = np.min(observed_losses)
    opt_arms.append(opt_arm)
    opt_losses.append(opt_loss)
    print('observed minimun loss is : '+ str(opt_loss))
  opt_losses = np.array(opt_losses)
  return opt_arms[np.argmin(opt_losses)], np.min(opt_losses)