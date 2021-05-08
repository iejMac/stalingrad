import numpy as np
from tensor import Tensor

np.random.seed(0)

X = [np.array([[1, 0, 1, 0]]),
     np.array([[0, 0, 1, 1]]),
     np.array([[1, 0, 0, 0]]),
     np.array([[0, 1, 0, 1]])]
Y = [10.0, 3.0, 8.0, 5.0]

w1 = Tensor(np.random.random((4, 5)), name = "w1")
w2 = Tensor(np.random.random((5, 1)), name = "w2")

epochs = 10
lr = 0.01

for e in range(epochs):
  for i in range(len(X)):
    x = Tensor(X[i], name = "x")
    y = Tensor(np.array([[Y[i]]]), name = "y")

    # Forward pass:
    h1 = x@w1
    h1.name = "h1"

    h2 = h1@w2
    h2.name = "h2"

    loss = (h2 - y)**2
    loss.name = "loss"
    
    # Backward pass:
    loss.backprop()

    # Apply grads:
    w1.data -= lr * w1.grad
    w2.data -= lr * w2.grad

    loss.zero_grad()

print(w1)
print(w2)
