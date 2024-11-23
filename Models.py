import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class BaseModel(nn.Module):
  '''
  I'll try to tune the 4 out_channels, 2 drop out probability and learning rate, std for initializing weights.
  '''

  def __init__(self, config):
    super().__init__()
    # net : stack of convolutional layers s.t. extract features
    self.config = config
    self.net = nn.Sequential(nn.Conv2d(in_channels = 1, out_channels = int(self.config['oc1']), kernel_size = 3, padding = 1), # 28
                             nn.ReLU(),
                             nn.BatchNorm2d(int(self.config['oc1'])),
                             nn.MaxPool2d(kernel_size = 3, stride = 2), # 14
                             nn.Conv2d(in_channels = int(self.config['oc1']), out_channels = int(self.config['oc2']), kernel_size = 3, padding = 1), #14
                             nn.ReLU(),
                             nn.BatchNorm2d(int(self.config['oc2'])),
                             nn.MaxPool2d(kernel_size = 3, stride = 2), # 6
                             nn.Conv2d(in_channels = int(self.config['oc2']), out_channels = int(self.config['oc3']), kernel_size = 3, padding = 1), # 6
                             nn.ReLU(),
                             nn.Conv2d(in_channels = int(self.config['oc3']), out_channels = int(self.config['oc4']), kernel_size = 3, stride = 2, padding = 1), # 3,
                             nn.ReLU()
                             )

    # classifier : fully-connected linear layers
    self.classifier = nn.Sequential(nn.Dropout(p=self.config['do1']),
                                    nn.Linear(in_features=(int(self.config['oc4']) * 3 * 3), out_features = 1024),
                                    nn.ReLU(),
                                    nn.Dropout(p=self.config['do2']),
                                    nn.Linear(in_features = 1024, out_features = 1024),
                                    nn.ReLU(),
                                    nn.Linear(in_features = 1024, out_features = 10),
                                    )
    self.init_bias()

  def init_bias(self):
    for layer in self.net:
      if isinstance(layer, nn.Conv2d):
        nn.init.normal_(layer.weight, mean=0, std=self.config['std'])
        nn.init.constant_(layer.bias, 0)
    return

  def feed_forward(self, x):
    out = self.net(x)
    out = out.view(-1,int(self.config['oc4']) * 3 * 3)
    return self.classifier(out)
  