import nn
import numpy as np
from stalingrad.tensor import Tensor

t2 = np.ones((3, 2, 5, 5))
t2[:, :, 1] *= 2
t2[:, :, 2] *= 3
t2[:, :, 3] *= 4
t2[:, :, 4] *= 5

t2[:, 1] *= 2

t2[1] *= 2
t2[2] *= 3


t1 = Tensor(t2)

test = t1[0, :, 3:, 3:]
test.backward()

print(t1.grad)
