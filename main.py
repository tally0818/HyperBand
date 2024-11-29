import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from HyperBand import *
from RandomSearch import RandomSearch
import matplotlib.pyplot as plt
from Models import BaseModel


mnist_train = datasets.MNIST(root="../Data/", train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
mnist_test = datasets.MNIST(root="../Data/", train=False, transform=transforms.ToTensor(), target_transform=None, download=True)

train_loader = DataLoader(mnist_train, batch_size=100, shuffle=True, num_workers=2, drop_last=True)
test_loader = DataLoader(mnist_test, batch_size=100, shuffle=False, num_workers=2, drop_last=True)

space_config = {'lr' : {'distribution' : 'log_uniform',
                                       'min' : 1e-4,
                                       'max' : 1e-2},
                'do1' : {'distribution' : 'uniform',
                                 'min' : 0.3,
                                 'max' : 0.7},
                'do2' : {'distribution' : 'uniform',
                                       'min' : 0.3,
                                       'max' : 0.7},
                'oc1' : {'distribution' : 'uniform',
                                       'min' : 10,
                                       'max' : 20},
                'oc2' : {'distribution' : 'uniform',
                                       'min' : 20,
                                       'max' : 50},
                'oc3' : {'distribution' : 'uniform',
                                       'min' : 50,
                                       'max' : 200},
                'oc4' : {'distribution' : 'uniform',
                                       'min' : 200,
                                       'max' : 500},
                'std' : {'distribution' : 'log_uniform',
                                       'min' : 1e-4,
                                       'max' : 1e-1}
                }
search_space = SearchSpace(space_config)

R_values = list(range(15, 301, 15))
hyperband_best_losses = []
randomsearch_best_losses = []
randomsearch_2x_best_losses = []
bracket_best_losses = []

for R in R_values:

    hyperband_opt = HyperBand(R=R, eta=3)
    best_config_hb, best_loss_hb = hyperband_opt.optimize(BaseModel, search_space, train_loader, test_loader)
    hyperband_best_losses.append(best_loss_hb)

    randomsearch_opt = RandomSearch(R=R, eta=3)
    best_config_rs, best_loss_rs = randomsearch_opt.optimize(BaseModel, search_space, train_loader, test_loader)
    randomsearch_best_losses.append(best_loss_rs)

    randomsearch_2x_opt = RandomSearch(R=2*R, eta=3)
    best_config_rs_2x, best_loss_rs_2x = randomsearch_2x_opt.optimize(BaseModel, search_space, train_loader, test_loader)
    randomsearch_2x_best_losses.append(best_loss_rs_2x)

    bracket_opt =Bracket(R = R, eta = 3, s = 4)
    best_config_bracket_opt, best_loss_bracket_opt = bracket_opt.optimize(BaseModel, search_space, train_loader, test_loader)
    bracket_best_losses.append(best_loss_rs_2x)


plt.figure(figsize=(10, 6))
plt.plot(R_values, hyperband_best_losses, label='HyperBand')
plt.plot(R_values, randomsearch_best_losses, label='RandomSearch')
plt.plot(R_values, randomsearch_2x_best_losses, label='RandomSearch_2x')
plt.plot(R_values, bracket_best_losses, label='bracket(s = 4)')
plt.xlabel('R')
plt.ylabel('Best Loss')
plt.title('HyperBand vs. Others (eta = 3)')
plt.legend()
plt.grid(True)
plt.show()