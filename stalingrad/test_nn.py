import nn
import numpy as np
import functional as F
from tensor import Tensor

class TestNetwork(nn.Module):
  def __init__(self):
    super().__init__()

    self.l1 = nn.Dense(784, 100, use_bias=True)
    self.l2 = nn.Dense(100, 10, use_bias=True)

  def forward(self, x):
    x = self.l1(x)
    F.relu(x)
    x = self.l2(x)
    F.relu(x) # Needs to be softmax
    return x

x = Tensor(np.random.uniform(size=(1, 784)))
model = TestNetwork()
y = model(x)

print(model.parameters().keys())
