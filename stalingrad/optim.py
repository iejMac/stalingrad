import numpy as np

from stalingrad import Tensor

class Optimizer:
  def __init__(self, parameters):
    self.parameters = {name : param for name, param in parameters.items() if param.requires_grad}
  def zero_grad(self):
    for param in self.parameters.values():
      param.grad = Tensor(np.zeros(param.shape), device=param.device, requires_grad=False)

class SGD(Optimizer):
  def __init__(self, parameters, learning_rate=3e-4):
    super().__init__(parameters)
    self.learning_rate = learning_rate
  def step(self):
    for param in self.parameters.values():
      param -= self.learning_rate * param.grad

class AdaGrad(Optimizer):
  def __init__(self, parameters, learning_rate=3e-4):
    super().__init__(parameters)
    self.learning_rate = learning_rate
    self.accumulated_grads = dict([(key, Tensor(np.zeros(param.shape), device=param.device, requires_grad=False)) for key, param in self.parameters.items()])
  def step(self):
    for key, param in self.parameters.items():
      self.accumulated_grads[key] += param.grad ** 2
      param -= (self.learning_rate / (self.accumulated_grads[key]**0.5 + 1e-7)) * param.grad

class RMSProp(Optimizer):
  def __init__(self, parameters, learning_rate=3e-4, alpha=0.9):
    super().__init__(parameters)
    self.alpha = alpha
    self.learning_rate = learning_rate
    self.accumulated_grads = dict([(key, Tensor(np.zeros(param.shape), device=param.device, requires_grad=False) for key, param in self.parameters.items()])
  def step(self):
    for key, param in self.parameters.items():
      self.accumulated_grads[key] = self.alpha * self.accumulated_grads[key] + (1 - self.alpha) * (param.grad ** 2)
      param -= (self.learning_rate / ((self.accumulated_grads[key] + 1e-6)**0.5)) * param.grad

class Adam(Optimizer): # https://arxiv.org/pdf/1412.6980.pdf
  def __init__(self, parameters, learning_rate=3e-4, beta1=0.9, beta2=0.999):
    super().__init__(parameters)
    self.t = 0
    self.b1, self.b2 = beta1, beta2
    self.learning_rate = learning_rate
    self.acc_s = dict([(key, Tensor(np.zeros(param.shape), device=param.device, requires_grad=False)) for key, param in self.parameters.items()])
    self.acc_r = dict([(key, Tensor(np.zeros(param.shape), device=param.device, requires_grad=False)) for key, param in self.parameters.items()])
  def step(self):
    self.t += 1
    for key, param in self.parameters.items():
      self.acc_s[key] = self.b1 * self.acc_s[key] + (1 - self.b1) * param.grad
      self.acc_r[key] = self.b2 * self.acc_r[key] + (1 - self.b2) * (param.grad ** 2)
      s_ = self.acc_s[key] / (1 - self.b1**self.t)
      r_ = self.acc_r[key] / (1 - self.b2**self.t)
      param -= (self.learning_rate * s_) / (r_**0.5 + 1e-8)
