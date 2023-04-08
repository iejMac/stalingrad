"""MNIST functions"""
import numpy as np

# TODO: add download MNIST code
# mnist loader from : https://github.com/geohot/tinygrad/blob/master/test/test_mnist.py
def fetch_mnist(flatten=False, one_hot=False, data_dir="data/mnist"):
  import gzip
  parse = lambda file: np.frombuffer(gzip.open(file).read(), dtype=np.uint8).copy()
  shape = (-1, 28*28) if flatten else (-1, 28, 28)
  X_train = parse(f"{data_dir}/train-images-idx3-ubyte.gz")[0x10:].reshape(shape).astype(np.float32)
  Y_train = parse(f"{data_dir}/train-labels-idx1-ubyte.gz")[8:]
  X_test = parse(f"{data_dir}/t10k-images-idx3-ubyte.gz")[0x10:].reshape(shape).astype(np.float32)
  Y_test = parse(f"{data_dir}/t10k-labels-idx1-ubyte.gz")[8:]

  if one_hot:
    Y_train_onehot, Y_test_onehot = np.zeros((len(Y_train), 10)), np.zeros((len(Y_test), 10))
    rows_train, rows_test = np.arange(len(Y_train)), np.arange(len(Y_test))
    Y_train_onehot[rows_train, Y_train], Y_test_onehot[rows_test, Y_test] = 1.0, 1.0
    Y_train, Y_test = Y_train_onehot, Y_test_onehot

  return X_train, Y_train, X_test, Y_test
