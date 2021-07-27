import nn
import numpy as np
from stalingrad.tensor import Tensor


imgs = Tensor(np.ones((10, 1, 28, 28)), requires_grad=True)


conv1 = nn.Conv2d(1, 4, 3)
conv2 = nn.Conv2d(4, 8, 3)
conv3 = nn.Conv2d(8, 1, 3)

out = conv1(imgs).relu()
out = conv2(out).relu()
out = conv3(out).relu()

print(out.shape)

