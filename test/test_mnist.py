import unittest
import numpy as np

from stalingrad import nn
from stalingrad import optim
from stalingrad.tensor import Tensor
from stalingrad.utils import train_module, test_accuracy
from stalingrad.data import fetch_mnist

np.random.seed(80085)

X_train, Y_train, X_test, Y_test = fetch_mnist(flatten=True, one_hot=True)

class LinearMnistClassifier(nn.Module):
  def __init__(self):
    super().__init__()
    self.lin1 = nn.Linear(784, 100)
    self.lin2 = nn.Linear(100, 50)
    self.lin3 = nn.Linear(50, 10)

  def forward(self, x):
    x = self.lin1(x).relu()
    x = self.lin2(x).relu()
    x = self.lin3(x).softmax()
    return x

class ConvolutionalMnistClassifier(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1)
    self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1)
    self.conv3 = nn.Conv2d(16, 1, kernel_size=3, stride=1)
    self.lin1 = nn.Linear(484, 10)
  def forward(self, x):
    x = x.reshape(shape=(-1, 1, 28, 28))
    x = self.conv1(x).relu()
    x = self.conv2(x).relu()
    x = self.conv3(x).relu()
    x = x.reshape(shape=(x.shape[0], -1))
    x = self.lin1(x).softmax()
    return x

class TestMNIST(unittest.TestCase):
  def test_linear_mnist(self):
    mod = LinearMnistClassifier()
    opt = optim.SGD(mod.parameters(), learning_rate=1e-2)
    loss_func = nn.NLL(reduction="mean")

    train_module(mod, opt, loss_func, X_train, Y_train, steps=500, batch_size=200)
    correct_pct = test_accuracy(mod, X_test, Y_test)
    print(f"Linear MNIST Classifier test accuracy: {correct_pct}")
    self.assertTrue(correct_pct > 0.95)

  def test_convolutional_mnist(self):
    mod = ConvolutionalMnistClassifier()
    opt = optim.Adam(mod.parameters(), learning_rate=1e-3)
    loss_func = nn.NLL(reduction="mean")

    train_module(mod, opt, loss_func, X_train, Y_train, steps=500, batch_size=200)
    correct_pct = test_accuracy(mod, X_test, Y_test)
    print(f"Convolutional MNIST Classifier test accuracy: {correct_pct}")
    self.assertTrue(correct_pct > 0.95)

if __name__ == "__main__":
  unittest.main()
