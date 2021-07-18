import numpy as np
from stalingrad.tensor import Tensor

# -= Modules =-

class Module:
  def __call__(self, x):
    return self.forward(x)
  def parameters(self, parent="root"):
    params = {}
    for attr in self.__dict__:
      if isinstance(self.__dict__[attr], Tensor):
        params[parent + "." + attr] = self.__dict__[attr]
      elif isinstance(self.__dict__[attr], Module):
        params.update(self.__dict__[attr].parameters(parent + "." + attr))
    return params
  
class Linear(Module):
  def __init__(self, in_neurons, out_neurons, use_bias=True):
    self.use_bias = use_bias
    self.shape = (in_neurons, out_neurons)

    self.weight = Tensor(np.random.uniform(-1., 1., size=self.shape)/np.sqrt(np.prod(self.shape)).astype(np.float32))
    self.bias = None
    if use_bias:
      self.bias = Tensor(np.random.uniform(-1., 1., (1, out_neurons)))

  def forward(self, x):
    result = x @ self.weight
    if self.use_bias:
      result += self.bias
    return result

# -= Losses =-

class Loss(Module):
  def __init__(self, reduction=None, reduce_axis=0):
    self.reduction=reduction
    self.reduce_axis = reduce_axis
  def __call__(self, prediction, target):
    loss = self.forward(prediction, target)
    if self.reduction is not None:
      loss = getattr(loss, self.reduction)(axis=self.reduce_axis)
    return loss

class MSE(Loss):
  def __init__(self, reduction=None, reduce_axis=0):
    super().__init__(reduction, reduce_axis)
  def forward(self, prediction, target):
    return (prediction - target)**2

class NLL(Loss):
  def __init__(self, reduction=None, reduce_axis=0):
    super().__init__(reduction, reduce_axis)
  def forward(self, prediction, target):
    sum_axis = list(range(len(prediction.shape)))
    sum_axis.remove(self.reduce_axis)
    return (target * prediction.log() * (-1.0)).sum(axis=tuple(sum_axis))
