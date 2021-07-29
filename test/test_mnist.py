import unittest
import numpy as np

from stalingrad import nn
from stalingrad import optim
from stalingrad.tensor import Tensor

np.random.seed(80085)

# mnist loader from : https://github.com/geohot/tinygrad/blob/master/test/test_mnist.py
def fetch_mnist(flatten=False, one_hot=False):
  import gzip
  parse = lambda file: np.frombuffer(gzip.open(file).read(), dtype=np.uint8).copy()
  shape = (-1, 28*28) if flatten else (-1, 28, 28)
  X_train = parse("test/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape(shape).astype(np.float32)
  Y_train = parse("test/mnist/train-labels-idx1-ubyte.gz")[8:]
  X_test = parse("test/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape(shape).astype(np.float32)
  Y_test = parse("test/mnist/t10k-labels-idx1-ubyte.gz")[8:]

  if one_hot:
    Y_train_onehot, Y_test_onehot = np.zeros((len(Y_train), 10)), np.zeros((len(Y_test), 10))
    rows_train, rows_test = np.arange(len(Y_train)), np.arange(len(Y_test))
    Y_train_onehot[rows_train, Y_train], Y_test_onehot[rows_test, Y_test] = 1.0, 1.0
    Y_train, Y_test = Y_train_onehot, Y_test_onehot

  return X_train, Y_train, X_test, Y_test

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
    self.conv1 = nn.Conv2d(1, 8, kernel_size=5, stride=1)
    self.conv2 = nn.Conv2d(8, 16, kernel_size=5, stride=1)
    self.conv3 = nn.Conv2d(16, 1, kernel_size=5, stride=1)
    self.lin1 = nn.Linear(256, 10)
  def forward(self, x):
    x = self.conv1(x).relu()
    x = self.conv2(x).relu()
    x = self.conv3(x).relu()
    x = x.reshape(shape=(x.shape[0], -1))
    x = self.lin1(x).softmax()
    return x

class TestMNIST(unittest.TestCase):
  def test_linear_mnist(self):
    steps = 500
    batch_size = 200
    lr = 1e-2

    mnist_classifier = LinearMnistClassifier()
    opt = optim.SGD(mnist_classifier.parameters(), learning_rate=lr)
    loss_func = nn.NLL(reduction="mean")
    X_train, Y_train, X_test, Y_test = fetch_mnist(flatten=True, one_hot=True)

    # train:
    for step in range(steps):
      ind = np.random.randint(0, len(X_train), size=(batch_size))
      X_batch = Tensor(X_train[ind], requires_grad=False)
      Y_batch = Tensor(Y_train[ind], requires_grad=False)

      probs = mnist_classifier(X_batch)
      loss = loss_func(probs, Y_batch)

      loss.backward()
      opt.step()
      opt.zero_grad()

    #evaluate:
    preds = mnist_classifier(Tensor(X_test, requires_grad=False))
    correct = 0

    for i in range(len(Y_test)):
      lab, pred = np.argmax(Y_test[i]), np.argmax(preds[i].data)
      correct += (lab == pred)
    
    correct_pct = correct/len(Y_test)
    print(f"Linear MNIST Classifier test accuracy: {correct_pct}")
    self.assertTrue(correct_pct > 0.95)

  def test_convolutional_mnist(self):
    steps = 100
    batch_size = 200
    lr = 1e-3

    mnist_classifier = ConvolutionalMnistClassifier()
    opt = optim.AdaGrad(mnist_classifier.parameters(), learning_rate=lr)
    loss_func = nn.NLL(reduction="mean")
    X_train, Y_train, X_test, Y_test = fetch_mnist(flatten=False, one_hot=True)
    X_train = np.expand_dims(X_train, 1)
    X_test = np.expand_dims(X_test, 1)

    # train:
    for step in range(steps):
      ind = np.random.randint(0, len(X_train), size=(batch_size))
      X_batch = Tensor(X_train[ind], requires_grad=False)
      Y_batch = Tensor(Y_train[ind], requires_grad=False)

      probs = mnist_classifier(X_batch)
      loss = loss_func(probs, Y_batch)
      print(loss)

      loss.backward()
      opt.step()
      opt.zero_grad()

    #evaluate:
    preds = mnist_classifier(Tensor(X_test, requires_grad=False))
    correct = 0

    for i in range(len(Y_test)):
      lab, pred = np.argmax(Y_test[i]), np.argmax(preds[i].data)
      correct += (lab == pred)
    
    correct_pct = correct/len(Y_test)
    print(f"Convolutional MNIST Classifier test accuracy: {correct_pct}")
    self.assertTrue(correct_pct > 0.95)

if __name__ == "__main__":
  unittest.main()
