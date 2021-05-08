import random
from stalingrad.variable import Variable

# Trying to make fake neural net

def matmul(net, x):
  result = Variable(0)
  for i, neuron in enumerate(net):
    result += neuron[0] * x[i] + neuron[1]
  return result

neuron_count = 4
lr = 0.1
epochs = 20

net = [[Variable(random.random(), f"weight_{i}"), Variable(random.random(), f"bias_{i}")] for i in range(neuron_count)]

X = [[1, 1, 0, 0],
     [1, 0, 0, 1],
     [0, 1, 0, 1],
     [1, 1, 1, 1],
     [0, 0, 0, 1],
     [0, 0, 1, 0]]

y = [12, 9, 5, 15, 1, 2]

for e in range(epochs):
  for i, x in enumerate(X):
    pred = matmul(net, x)

    mse_loss = (pred + (-y[i]))**2
    print(mse_loss.value)
    mse_loss.backprop()

    # Apply grad:
    for neuron in net:
      neuron[0].value -= neuron[0].grad * lr
      neuron[1].value -= neuron[1].grad * lr

    mse_loss.zero_grad()

# Weights should be: 8, 4, 2, 1
for neuron in net:
  print(neuron[0].value, neuron[1].value)
