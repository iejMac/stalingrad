import numpy as np
import nn
from stalingrad.tensor import Tensor

imgs = Tensor(np.ones((5, 2, 11, 11)), requires_grad=True)
kern = Tensor(np.ones((4, 1, 3, 3)), requires_grad=True)

out = imgs.conv2d(kern, groups=2)

out.backward()
