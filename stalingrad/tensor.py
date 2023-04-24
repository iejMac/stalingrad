import os

import inspect
import numpy as np

from collections import defaultdict


class Device:
  # gathers information about device ops
  _ops = sorted(os.listdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), "ops")))
  imports = dict(enumerate([os.path.splitext(x)[0] for x in _ops if x.startswith("ops_")]))
  buffers = {}
  for i,op in imports.items():
    name = op[len("ops_"):].upper()
    vars()[name] = i
  DEFAULT = CPU


class Tensor:
  ops = defaultdict(dict)

  def __init__(self, data, requires_grad=True, name="", device=Device.DEFAULT):
    self.data, self.device = self._move_data(data, device)
    self.name = name
    self.requires_grad = requires_grad
    self.grad = Tensor(np.zeros(self.shape, dtype=np.float32), device=device, requires_grad=False) if requires_grad else None
    self.func = None # Function that created the Tensor

  @staticmethod
  def _move_data(data, device):
    if isinstance(device, str):
      dev_ind = device.split(":")
      # TODO: ind will be used to specify which GPU in the future
      dev_type, ind = (dev_ind[0], int(dev_ind[1])) if len(dev_ind) > 1 else (dev_ind[0], 0)
      device = getattr(Device, dev_type.upper())

    if isinstance(data, np.ndarray):
      data = data.view(Device.buffers[Device.CPU])
    elif isinstance(data, Device.buffers[device]):
      return data, device

    data = data.toCPU().view(Device.buffers[Device.CPU])
    return Device.buffers[device].fromCPU(data), device

  def to(self, device):
    self.data, self.device = self._move_data(self.data, device)
    if self.requires_grad:
      self.grad.to(device)
    return

  @property
  def shape(self):
    return self.data.shape
  @property
  def dtype(self):
    return self.data.dtype
  def __repr__(self):
    return self.data.__repr__()
  def assign(self, x):
    self.data = x.data
    return self

  def __getitem__(self, slices):
    return self.slice(inds=tuple([slices]) if isinstance(slices, (int, slice)) else slices)
  def __setitem__(self, slices, value):
    self.data[slices] = value

  def backward(self, passed_grad=None):
    if self.func is None:
      return

    if passed_grad is None: # root call
      self.grad += Tensor(np.ones(self.shape, dtype=self.dtype), device=self.device, requires_grad=False) # df/df = 1
      passed_grad = self.grad

    grads = self.func.backward(self.func, passed_grad.data)
    grads = grads if len(self.func.parents) > 1 else [grads]
    grads = [Tensor(g, device=self.device, requires_grad=False) for g in grads]

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
                 requires_grad=any([t.requires_grad for t in x]), device=func.device)
    if ret.requires_grad:
      ret.func = func
    return ret
    
def register_operations(name, func, device):
  Tensor.ops[device][name] = func
  def compute(*x, **kwargs):
    tsr = [arg for arg in x if isinstance(arg, Tensor)][0] # first tensor in args
    x = [Tensor(np.array([arg], dtype=np.float32), requires_grad=False, device=tsr.device) if not isinstance(arg, Tensor) else arg for arg in x]
    f = Tensor.ops[tsr.device][name]
    f.device = tsr.device
    return f.apply(f, *x, **kwargs)
  setattr(Tensor, name, compute)
  if name in ["add", "sub", "mul", "matmul", "pow"]:
    setattr(Tensor, f"__{name}__", compute)
    setattr(Tensor, f"__i{name}__", lambda self, x: self.assign(compute(self, x)))
    setattr(Tensor, f"__r{name}__", lambda self, x: compute(x, self))

def _register_operations(namespace, device):
  for name, cls in inspect.getmembers(namespace, inspect.isclass):
    if name.endswith("Buffer"):
      Device.buffers[device] = cls
    elif name != "Function":
      register_operations(name.lower(), cls, device)


import importlib
for d,ops in Device.imports.items():
  try:
    _register_operations(importlib.import_module('stalingrad.ops.'+ops), d)
  except ImportError:
    pass
