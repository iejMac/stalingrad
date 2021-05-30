import unittest
import numpy as np
from stalingrad.tensor import Tensor

np.random.seed(0)

X_train = np.array([[[1, 0, 1, 0]],
                   [[0, 0, 1, 1]],
                   [[1, 0, 0, 0]],
                   [[0, 1, 0, 1]]])
X_test = np.array([[[1, 1, 1, 0]],
                   [[0, 0, 0, 1]],
                   [[1, 1, 0, 0]],
                   [[0, 0, 1, 0]]])
Y_train = np.array([10.0, 3.0, 8.0, 5.0])
Y_test = np.array([14.0, 1.0, 12.0, 2.0])

class TestTensor(unittest.TestCase):
  def test_tensor(self):
    w1 = Tensor(np.random.random((4, 5)), name = "w1")
    w2 = Tensor(np.random.random((5, 1)), name = "w2")

    epochs = 100
    lr = 0.01

    # Train:
    for e in range(epochs):
      for i in range(len(X_train)):
        x = Tensor(X_train[i], name = "x")
        y = Tensor(np.array([[Y_train[i]]]), name = "y")

        # Forward pass:
        h1 = x @ w1
        h2 = h1 @ w2
        loss = (h2 - y)**2

        # Backward pass:
        loss.backprop()

        # Apply grads:
        w1.data -= lr * w1.grad
        w2.data -= lr * w2.grad

        loss.zero_grad()
        x.zero_grad()
        y.zero_grad()
        w1.zero_grad()
        w2.zero_grad()
        h1.zero_grad()
        h2.zero_grad()

    # Test:
    cumulative_loss = 0.0

    for i, x in enumerate(X_test):
      x = Tensor(x, name="x")
      y = Tensor(Y_test[i], name="y")

      h1 = x @ w1
      h2 = h1 @ w2
      loss = (h2 - y)**2
      print(loss)

      cumulative_loss += loss.data

    self.assertTrue(np.all(cumulative_loss < 5e-2))

if __name__ == "__main__":
  unittest.main()
