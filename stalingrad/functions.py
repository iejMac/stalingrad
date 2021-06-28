import numpy as np
from tensor import Function

class ReLU(Function):
  def forward(func, x):
    func.save_tensors(x)
    return np.maximum(x, 0)
  def backward(func, passed_grad):
    x = func.saved_tensors[0]
    return passed_grad * (x >= 0)

class Add(Function):
  def forward(func, x, y):
    return x+y
  def backward(func, passed_grad):
    return passed_grad, passed_grad

class Sub(Function):
  def forward(func, x, y):
    return x-y
  def backward(func, passed_grad):
    return passed_grad, -passed_grad

class Mul(Function):
  def forward(func, x, y):
    func.save_tensors(x, y)
    return x*y
  def backward(func, passed_grad):
    x, y = func.saved_tensors
    return y*passed_grad, x*passed_grad

class Pow(Function):
  def forward(func, x, y):
    func.save_tensors(x, y)
    return x**y
  def backward(func, passed_grad):
    x, y = func.saved_tensors
    return y * (x**(y-1.0)) * passed_grad, (x**y) * np.log(x) * passed_grad

class Matmul(Function):
  def forward(func, x, y):
    func.save_tensors(x, y)
    return x @ y
  def backward(func, passed_grad):
    x, y = func.saved_tensors
    return passed_grad @ np.swapaxes(y, -2, -1), np.swapaxes(x, -2, -1) @ passed_grad
