import nn
import numpy as np
from stalingrad.tensor import Tensor

imgs = Tensor(np.ones((5, 1, 10, 10)))

k = np.ones((4, 1, 3, 3))
k[:, :, 1] *= -1
k[:, :, 0, 2] *= -1

print(k)

kern = Tensor(k)

out = imgs.convtranspose2d(kern, stride=(2, 2))
r = out.relu()

r.backward()

print(kern.grad)
print(imgs.grad)



out.backward()

