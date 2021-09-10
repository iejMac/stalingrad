import inspect
import numpy as np

from collections import defaultdict

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

  def __getitem__(self, slices):
    return self.slice(inds=tuple([slices]) if isinstance(slices, (int, slice)) else slices)

  def backward(self, passed_grad=None):
    if self.func is None:
      return

    if passed_grad is None: # root call
      self.grad += np.ones(self.shape, dtype=self.dtype) # df/df = 1
      passed_grad = self.grad

    grads = self.func.backward(self.func, passed_grad)
    grads = grads if len(self.func.parents) > 1 else [grads]

    for p, g in zip(self.func.parents, grads):
      if p.requires_grad:
        p.grad += g
        p.backward(g)

  def div(self, x):
    return self * (x**(-1.0))
  __truediv__ = div
  def mean(self, axis=None):
    sm = self.sum(axis=axis)
    mean = sm * (np.prod(sm.shape) / np.prod(self.shape))
    return mean
  def sigmoid(self):
    e_x = self.exp()
    return e_x / (e_x + 1)
  def softmax(self, dist_axes=(1,)):
    axis = (dist_axes,) if isinstance(dist_axes, int) else dist_axes
    shape = [1 if ax in axis else self.shape[ax] for ax in range(len(self.shape))]
    e_x = self.exp()
    return e_x / (e_x.sum(axis=dist_axes).reshape(shape=shape))
  def tanh(self):
    return (2.0 * self).sigmoid() * 2.0 - 1.0


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

  def apply(self, *x, **kwargs):
    func = self(*x)
    ret = Tensor(self.forward(func, *[t.data for t in x], **kwargs),
                 requires_grad=any([t.requires_grad for t in x]))
    if ret.requires_grad:
      ret.func = func
    return ret
    
def register_operations(name, func):
  def compute(*x, **kwargs):
    x = [Tensor(np.array([arg]), requires_grad=False) if not isinstance(arg, Tensor) else arg for arg in x]
    return func.apply(func, *x, **kwargs)
  setattr(Tensor, name, compute)
  if name in ["add", "sub", "mul", "matmul", "pow"]:
    setattr(Tensor, f"__{name}__", compute)
    setattr(Tensor, f"__r{name}__", lambda self, x: compute(x, self))

def _register_operations(namespace):
  for name, cls in inspect.getmembers(namespace, inspect.isclass):
    if name != "Function":
      register_operations(name.lower(), cls)
      
# import functions
from stalingrad import functions
_register_operations(functions)
