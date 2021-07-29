import numpy as np

# Make more general Optimizer class and have all specific ones inherit
class Optimizer:
  def __init__(self, parameters):
    self.parameters = {name : param for name, param in parameters.items() if param.requires_grad}
  def zero_grad(self):
    for param in self.parameters.values():
      param.grad = np.zeros(param.shape)

class SGD(Optimizer):
  def __init__(self, parameters, learning_rate=3e-4):
    super().__init__(parameters)
    self.learning_rate = learning_rate
  def step(self):
    for param in self.parameters.values():
      param.data -= self.learning_rate * param.grad

class AdaGrad(Optimizer):
  def __init__(self, parameters, learning_rate=3e-4):
    super().__init__(parameters)
    self.learning_rate = learning_rate
    self.accumulated_squared_grads = dict([(key, np.zeros(param.shape)) for key, param in self.parameters.items()])
  def step(self):
    for key, param in self.parameters.items():
      self.accumulated_squared_grads[key] += param.grad ** 2
      param.data -= (self.learning_rate / (self.accumulated_squared_grads[key]**0.5 + 1e-7)) * param.grad
