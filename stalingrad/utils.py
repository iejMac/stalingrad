import pickle
import numpy as np
from stalingrad.tensor import Tensor

def train_module(module, optimizer, loss_func, X_train, Y_train, steps=500, batch_size=200):
  for step in range(steps):
    ind = np.random.randint(0, len(X_train), size=(batch_size))
    X_batch = Tensor(X_train[ind], requires_grad=False)
    Y_batch = Tensor(Y_train[ind], requires_grad=False)

    probs = module(X_batch)
    loss = loss_func(probs, Y_batch)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

def test_accuracy(module, X_test, Y_test):
  preds = module(Tensor(X_test, requires_grad=False))
  correct = 0
  for i in range(len(Y_test)):
    lab, pred = np.argmax(Y_test[i]), np.argmax(preds[i].data)
    correct += (lab == pred)
  return correct/len(Y_test)

# mnist loader from : https://github.com/geohot/tinygrad/blob/master/test/test_mnist.py
def fetch_mnist(flatten=False, one_hot=False):
  import gzip
  parse = lambda file: np.frombuffer(gzip.open(file).read(), dtype=np.uint8).copy()
  shape = (-1, 28*28) if flatten else (-1, 28, 28)
  X_train = parse("data/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape(shape).astype(np.float32)
  Y_train = parse("data/mnist/train-labels-idx1-ubyte.gz")[8:]
  X_test = parse("data/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape(shape).astype(np.float32)
  Y_test = parse("data/mnist/t10k-labels-idx1-ubyte.gz")[8:]

  if one_hot:
    Y_train_onehot, Y_test_onehot = np.zeros((len(Y_train), 10)), np.zeros((len(Y_test), 10))
    rows_train, rows_test = np.arange(len(Y_train)), np.arange(len(Y_test))
    Y_train_onehot[rows_train, Y_train], Y_test_onehot[rows_test, Y_test] = 1.0, 1.0
    Y_train, Y_test = Y_train_onehot, Y_test_onehot

  return X_train, Y_train, X_test, Y_test

def save_module(module, path):
  with open(path, "wb") as mod:
    pickle.dump(module, mod)

def load_module(path):
  with open(path, "rb") as mod:
    module = pickle.load(mod)
  return module
