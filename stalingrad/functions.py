import numpy as np
from stalingrad.tensor import Function

class ReLU(Function):
  def forward(func, x):
    func.save_tensors(x)
    return np.maximum(x, 0)
  def backward(func, passed_grad):
    x = func.saved_tensors[0]
    return passed_grad * (x >= 0)

class Log(Function):
  def forward(func, x):
    func.save_tensors(x)
    return np.log(x)
  def backward(func, passed_grad):
    x = func.saved_tensors[0]
    return passed_grad * 1/x

class Exp(Function):
  def forward(func, x):
    ret = np.exp(x)
    func.save_tensors(ret) # d(e^x)/dx = e^x
    return ret
  def backward(func, passed_grad):
    e_x = func.saved_tensors[0]
    return passed_grad * e_x

def unbroadcast(out, in_sh): # https://github.com/geohot/tinygrad/blob/master/tinygrad/ops_cpu.py (line 65)
  # It's possible to perform operations on tensors of different size f.e. Add((5, 5), (1, 5)) but for the backward
  # pass we need to remember to unbroadcast the output back to (1, 5)
  sum_axis = tuple([i for i in range(len(in_sh)) if in_sh[i]==1 and out.shape[i]>1]) if in_sh != (1,) else None
  return out.sum(axis=sum_axis).reshape(in_sh)

class Add(Function):
  def forward(func, x, y):
    func.save_tensors(x, y)
    return x+y
  def backward(func, passed_grad):
    x, y = func.saved_tensors
    return unbroadcast(passed_grad, x.shape), unbroadcast(passed_grad, y.shape)

class Sub(Function):
  def forward(func, x, y):
    func.save_tensors(x, y)
    return x-y
  def backward(func, passed_grad):
    x, y = func.saved_tensors
    return unbroadcast(passed_grad, x.shape), unbroadcast(-passed_grad, y.shape)

class Mul(Function):
  def forward(func, x, y):
    func.save_tensors(x, y)
    return x*y
  def backward(func, passed_grad):
    x, y = func.saved_tensors
    return unbroadcast(y*passed_grad, x.shape), unbroadcast(x*passed_grad, y.shape)

class Pow(Function):
  def forward(func, x, y):
    func.save_tensors(x, y)
    return x**y
  def backward(func, passed_grad):
    x, y = func.saved_tensors
    return unbroadcast(y * (x**(y-1.0)) * passed_grad, x.shape), unbroadcast((x**y) * np.log(x) * passed_grad, y.shape)

class Matmul(Function):
  def forward(func, x, y):
    func.save_tensors(x, y)
    return x @ y
  def backward(func, passed_grad):
    x, y = func.saved_tensors
    return passed_grad @ np.swapaxes(y, -2, -1), np.swapaxes(x, -2, -1) @ passed_grad

class Sum(Function):
  def forward(func, x, axis=None):
    ax = (axis,) if isinstance(axis, int) else axis
    func.save_tensors(x, ax)
    return x.sum(axis=ax) if ax is not None else np.array([x.sum()])
  def backward(func, passed_grad):
    x, axis = func.saved_tensors
    shape = [1 if (axis is None or ax in axis) else x.shape[ax] for ax in range(len(x.shape))]
    return passed_grad.reshape(shape) + np.zeros_like(x)

class Reshape(Function):
  def forward(func, x, shape=None):
    func.save_tensors(x, shape)
    return x.reshape(shape)
  def backward(func, passed_grad):
    x, _ = func.saved_tensors
    return passed_grad.reshape(x.shape)
