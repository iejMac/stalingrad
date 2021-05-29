import numpy as np
from tensor import Tensor

class Module:
  def __init__(self):
    pass
  def __call__(self, x):
    return self.forward(x)
  def forward(self, x):
    return x
  def parameters(self):
    params = {}
    for attr in self.__dict__:
      if isinstance(self.__dict__[attr], Tensor):
        params[attr] = self.__dict__[attr]
      elif isinstance(self.__dict__[attr], Module):
        params.update(self.__dict__[attr].parameters())
    return params
  
class Dense(Module):
  def __init__(self, in_neurons, out_neurons, use_bias=True):
    self.use_bias = use_bias
    self.shape = (in_neurons, out_neurons)

    self.weight = Tensor(np.random.uniform(-1., 1., size=self.shape)/np.sqrt(np.prod(self.shape)).astype(np.float32))
    self.bias = None
    if use_bias:
      self.bias = Tensor(np.random.uniform(-1., 1., (out_neurons, 1)))

  def forward(self, x):
    result = x @ self.weight
    if self.use_bias:
      result += self.bias
    return result
