import numpy as np

# Make more general Optimizer class and have all specific ones inherit
'''
class Optimizer:
  def __init__(self):
    pass
'''

class SGD:
  def __init__(self, parameters, learning_rate=3e-4):
    self.parameters = parameters
    self.learning_rate = learning_rate

  def step(self):
    for param in self.parameters:
      param.data -= self.learning_rate * param.grad
