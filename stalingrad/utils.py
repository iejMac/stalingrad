import numpy as np
from stalingrad.tensor import Tensor

def train_module(module, optimizer, loss_func, X_train, Y_train, X_test, Y_test, steps=500, batch_size=200):
  # train:
  for step in range(steps):
    ind = np.random.randint(0, len(X_train), size=(batch_size))
    X_batch = Tensor(X_train[ind], requires_grad=False)
    Y_batch = Tensor(Y_train[ind], requires_grad=False)

    probs = module(X_batch)
    loss = loss_func(probs, Y_batch)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

  #evaluate:
  preds = module(Tensor(X_test, requires_grad=False))
  correct = 0

  for i in range(len(Y_test)):
    lab, pred = np.argmax(Y_test[i]), np.argmax(preds[i].data)
    correct += (lab == pred)

  return correct/len(Y_test)
