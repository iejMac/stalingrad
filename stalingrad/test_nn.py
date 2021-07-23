import numpy as np
import nn
from stalingrad.tensor import Tensor

imgs = Tensor(np.ones((5, 10, 10)) * (np.arange(5) + 1).reshape(5, 1, 1), requires_grad=False)
fil = Tensor(np.ones((3, 3, 3)) * (np.arange(3) + 1).reshape(3, 1, 1), requires_grad=False)

out = imgs.conv2d(fil, padding=[(0, 1), (0, 1)])

print(imgs)
print(fil)

print(out)



