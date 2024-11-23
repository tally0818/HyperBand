from search_space import SearchSpace
from HyperBand import HyperBand
from Models import BaseModel
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


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

opt_config = HyperBand(search_space, 5, 3, train_loader, test_loader)
print(opt_config)