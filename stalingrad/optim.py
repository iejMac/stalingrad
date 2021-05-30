import numpy as np

# Make more general Optimizer class and have all specific ones inherit
class Optimizer:
  def __init__(self, parameters):
    self.parameters = [x for x in parameters if x.requires_grad]

  def zero_grad(self):
    for param in self.parameters:
      param.zero_grad()

class SGD(Optimizer):
  def __init__(self, parameters, learning_rate=3e-4):
    super().__init__(parameters)
    self.learning_rate = learning_rate

  def step(self):
    for param in self.parameters:
      param.data -= self.learning_rate * param.grad
