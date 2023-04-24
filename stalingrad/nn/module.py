"""base Module class"""

from stalingrad import Tensor

class Module:
  def __init__(self):
    self.training = True
  def __call__(self, *x):
    return self.forward(*x)
  def parameters(self, parent="root"):
    params = {}
    for attr in self.__dict__:
      if isinstance(self.__dict__[attr], Tensor):
        params[parent + "." + attr] = self.__dict__[attr]
      elif isinstance(self.__dict__[attr], Module):
        params.update(self.__dict__[attr].parameters(parent + "." + attr))
    return params
  def to(self, device):
    params = self.parameters()
    for name, p in params.items():
      p.to(device)
  def training(self):
    self.training = True
  def eval(self):
    self.training = False
