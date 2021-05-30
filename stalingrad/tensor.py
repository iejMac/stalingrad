import numpy as np

# Tensor version of Variable
class Tensor:
  def __init__(self, data, local_grads={}, requires_grad=True, name=""):
    self.data = data
    self.name = name
    self.local_grads = local_grads
    self.requires_grad = requires_grad
    self.grad = np.zeros(self.shape) if requires_grad else None

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
    
    if self.requires_grad:
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

def neg(x: Tensor) -> Tensor:
  return Tensor((-1.0) * x.data, {
    x: (-1.0) * (x.data != 0.0).astype(np.float32)
  })
def add(x: Tensor, y: Tensor) -> Tensor:
  return Tensor(x.data + y.data, {
    x: (x.data != 0.0).astype(np.float32),
    y: (y.data != 0.0).astype(np.float32)
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
