import nn
import numpy as np
from stalingrad.tensor import Tensor

test = Tensor(np.ones((3, 3, 3)))
t2= test.pad((2, 1))

t3 = test*2

t3.backward()


print(test.grad)


