import numpy as np
from stalingrad.tensor import Tensor

def relu(x: Tensor):
  x.data = np.maximum(x.data, 0.0)
