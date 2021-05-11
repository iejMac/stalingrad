import numpy as np
from stalingrad.tensor import Tensor

def relu(x: Tensor):
  x.data = np.maximum(x, 0.0)
