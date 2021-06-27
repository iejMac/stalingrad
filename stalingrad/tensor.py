import inspect
import numpy as np

from collections import defaultdict

# Tensor version of Variable
class Tensor:
  def __init__(self, data, requires_grad=True, name=""):
    self.data = data
    self.name = name
    self.requires_grad = requires_grad
    self.grad = np.zeros(self.shape) if requires_grad else None
    self.func = None # Function that created the Tensor

  @property
  def shape(self):
    return self.data.shape
  @property
  def dtype(self):
    return self.data.dtype
  def __repr__(self):
    return np.array_repr(self.data).replace("array", "Tensor")
  def assign(self, x):
    self.data = x.data

  def backward(self, passed_grad=None):
    if self.func is None:
      return

    if passed_grad is None: # root call
      self.grad += np.ones(self.shape, dtype=self.dtype) # df/df = 1
      passed_grad = self.grad

    grads = self.func.backward(self.func, passed_grad)
    grads = grads if len(self.func.parents) > 1 else [grads]
    
    for p, g in zip(self.func.parents, grads):
      p.grad += g
      p.backward(passed_grad)

class Function:
  def __new__(cls, *args, **kwargs):
    cls.forward = staticmethod(cls.forward)
    cls.backward = staticmethod(cls.backward)
    return super().__new__(cls)

  def __init__(self, *tensors):
    self.parents = tensors
    self.saved_tensors = []

  def save_tensors(self, *tensors):
    self.saved_tensors.extend(tensors)

  def apply(self, *x):
    func = self(*x)
    ret = Tensor(self.forward(func, *[t.data for t in x]),
                 requires_grad=any([t.requires_grad for t in x]))
    if ret.requires_grad:
      ret.func = func
    return ret
    
def register_operations(namespace):
  for name, cls in inspect.getmembers(namespace, inspect.isclass)[1:]:
    def compute(*x):
      return cls.apply(cls, *x)
    setattr(Tensor, name.lower(), compute)
   
import functions
register_operations(functions)
