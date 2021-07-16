import numpy as np
import nn
from stalingrad.tensor import Tensor

import optim

x = Tensor(np.random.uniform(size=(1, 20)), requires_grad=False)
labs = Tensor(np.ones(10).reshape((1, 10)), requires_grad=False)

d1 = nn.Linear(20, 10)
op = optim.SGD(d1.parameters(), 0.1)
y = d1(x)

# MSE = (y-labs)**2
log = y.log() * -1.0

# MSE.backward()
log.backward()

op.step()
op.zero_grad()
