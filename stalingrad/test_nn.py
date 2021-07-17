import numpy as np
import nn
from stalingrad.tensor import Tensor

import optim

x = Tensor(np.random.uniform(size=(1, 20)), requires_grad=False)
labs = Tensor(np.ones(10).reshape((1, 10)), requires_grad=False)

mse_loss = nn.MSE(reduction="mean", reduce_axis=None)

d1 = nn.Linear(20, 10)
op = optim.SGD(d1.parameters(), 0.1)
y = d1(x)

print(y)
print(labs)

loss = mse_loss(y, labs)
print(loss)
loss.backward()


op.step()
op.zero_grad()
