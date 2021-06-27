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
