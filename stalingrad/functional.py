import numpy as np
from tensor import Tensor

def relu(x: Tensor):
  x.data = np.maximum(x.data, 0.0)
