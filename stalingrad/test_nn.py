import numpy as np
import nn
from stalingrad.tensor import Tensor

import optim

x = Tensor(np.random.uniform(size=(2, 20)), requires_grad=False)
labs = Tensor(np.zeros((2, 10)), requires_grad=False)
labs.data[0][3] = 1.0
labs.data[1][5] = 1.0

nll_loss = nn.NLL(reduction="sum")

d1 = nn.Linear(20, 10)
op = optim.SGD(d1.parameters(), 0.1)
y = d1(x).softmax()

print(y)
print(labs)

loss = nll_loss(y, labs)
print(loss)
loss.backward()


op.step()
op.zero_grad()
