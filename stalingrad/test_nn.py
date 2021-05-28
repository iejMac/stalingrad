import nn
import numpy as np
import functional as F
from tensor import Tensor

x = Tensor(np.random.uniform(size=(1, 10)))
test = nn.Dense(10, 1)

y = test(x)
F.relu(y)

y.backprop()
