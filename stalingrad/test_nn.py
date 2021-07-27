import nn
import numpy as np
from stalingrad.tensor import Tensor

dat = np.ones((1, 1, 5, 5))
dat[0][0][1] *= 2
dat[0][0][2] *= 3
dat[0][0][3] *= 4
dat[0][0][4] *= 5

kern = np.ones((2, 1, 2, 2))
kern[0][0] *= 10
kern[1][0] *= -1

imgs = Tensor(dat, requires_grad=True)
kernels = Tensor(kern, requires_grad=True)

print(imgs)
print(kernels)

out = imgs.conv2d(kernels)
print(out)

out.backward()

print(imgs.grad)
print(kernels.grad)




