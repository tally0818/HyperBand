import numpy as np
from typing import Dict, Any, Union, List
class SearchSpace:
  '''


  '''
  def __init__(self, space_config : Dict[str, Dict[str, Any]]):
    '''
    Example of space_config

    space_config = {'learning_rate' : {'distribution' : 'log_uniform',
                                       'min' : 1e-4,
                                       'max' :1e-2},      
                    'dropout' : {'distribution' : 'uniform',
                                 'min' : 0.1,
                                 'max' : 0.7}, ...

                    }
    '''
    self.space_config = space_config
    self.validate_config()

  def validate_config(self):
    '''
    function that validates space_config, i.e. checks that space_configs has the right parameter for each distributuion
    at this moment, I implemented SearchSpace for Normal, uniform, log_uniform, and last beta distribution.
    '''
    for param_name, config in self.space_config.items():
      if 'distribution' not in config:
        raise ValueError(f"Distribution type not specified for parameter {param_name}")

      dist_type = config['distribution'].lower()
      if dist_type not in ['normal', 'uniform', 'log_uniform', 'beta']:
        raise ValueError(f"Unsupported distribution type {dist_type}")

      if dist_type == 'normal':
        required = ['mean', 'std']
      elif dist_type in ['uniform', 'log_uniform']:
        required = ['min', 'max']
      elif dist_type == 'beta':
        required = ['a', 'b']

      missing = [param for param in required if param not in config]
      if missing:
        raise ValueError(f"Missing required parameters {missing} for {dist_type} distribution")

  def sample(self, n_samples: int = 1):
    '''

    '''
    if n_samples < 1:
      raise ValueError("n_samples must be positive")

    sampling_methods = {'normal' : self._sample_normal,
                        'uniform' : self._sample_uniform,
                        'log_uniform' : self._sample_log_uniform,
                        'beta' : self._sample_beta}

    samples = []

    for _ in range(n_samples):
      sample = {}
      for param_name, config in self.space_config.items():
        dist_type = config['distribution'].lower()
        sample[param_name] = sampling_methods[dist_type](config)
        samples.append(sample)

    return samples

  def _sample_normal(self, config: Dict[str, Any]):
    return np.random.normal(config['mean'], config['std'])

  def _sample_uniform(self, config: Dict[str, Any]):
    return np.random.uniform(config['min'], config['max'])

  def _sample_log_uniform(self, config: Dict[str, Any]):
    log_min = np.log(config['min'])
    log_max = np.log(config['max'])
    return np.exp(np.random.uniform(log_min, log_max))

  def _sample_beta(self, config: Dict[str, Any]):
    return np.random.beta(config['a'], config['b'])

