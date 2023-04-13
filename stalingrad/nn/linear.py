""" """ #TODO: add docstring
import numpy as np

from stalingrad import Tensor
from stalingrad.nn import Module

class Linear(Module):
  def __init__(self, in_neurons, out_neurons, use_bias=True):
    super().__init__()
    self.use_bias = use_bias
    self.shape = (in_neurons, out_neurons)

    self.weight = Tensor(
      np.random.uniform(-1., 1., size=self.shape)/np.sqrt(np.prod(self.shape)).astype(np.float32),
      name="linear.weight"
    )
    self.bias = None
    if use_bias:
      self.bias = Tensor(
        np.random.uniform(-1., 1., (1, out_neurons)),
        name="linear.bias"
      )

  def forward(self, x):
    result = x @ self.weight
    if self.use_bias:
      result += self.bias
    return result
