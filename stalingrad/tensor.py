import numpy as np

# Tensor version of Variable
class Tensor:
  def __init__(self, data, local_grads={}, name=""):
    self.data = data
    self.name = name
    self.local_grads = local_grads
    self.grad = np.zeros(self.shape)

  @property
  def shape(self):
    return self.data.shape
  @property
  def dtype(self):
    return self.data.dtype
  def __repr__(self):
    #TODO: make this say Tensor(...) instead of array(...)
    return np.array_repr(self.data)

  def assign(self, x):
    self.data = x.data

  def __matmul__(self, x):
    return matmul(self, x)
  def __add__(self, x):
    return add(self, x)
  def __sub__(self, x):
    return add(self, neg(x))
  def __pow__(self, x):
    return pow(self, x)

  def backprop(self, root=True):
    if root == True:
      self.grad = np.ones(self.shape)

    # TODO: hack, swap order for matmul dim balancing
    swap = False
    for child_tensor in self.local_grads:
      if swap is False:
        child_tensor.grad += self.grad @ self.local_grads[child_tensor]
      else:
        child_tensor.grad += self.local_grads[child_tensor] @ self.grad
      child_tensor.backprop(False)
      swap = True

  def zero_grad(self):
    self.grad = np.zeros(self.shape)
    for child_tensor in self.local_grads:
      if np.any(child_tensor.grad != 0):
        child_tensor.zero_grad()

def neg(x: Tensor) -> Tensor:
  return Tensor((-1.0) * x.data, {
    x: (-1.0) * np.ones(x.shape)
  })
def add(x: Tensor, y: Tensor) -> Tensor:
  return Tensor(x.data + y.data, {
    x: np.ones(x.shape),
    y: np.ones(y.shape)
  })
def matmul(x: Tensor, y: Tensor) -> Tensor:
  return Tensor(np.matmul(x.data, y.data), {
    x: y.data.T,
    y: x.data.T
  })
def pow(x: Tensor, y) -> Tensor:
  return Tensor(x.data**y, {
    x: y * x.data**(y-1)
  })