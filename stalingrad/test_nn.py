import nn
import numpy as np
from stalingrad.tensor import Tensor

imgs = Tensor(np.ones((5, 2, 10, 10)))

conv = nn.ConvTranspose2d(5, 4, 3, groups=2)

out = conv(imgs)
out.backward()

print(out.shape)

