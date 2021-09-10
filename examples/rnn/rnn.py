import numpy as np

from stalingrad import nn
from stalingrad import optim
from stalingrad.tensor import Tensor


class RNN(nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim):
    super().__init__()
    self.ixh = nn.Linear(input_dim, hidden_dim)
    self.hxh = nn.Linear(hidden_dim, hidden_dim)
    self.hxo = nn.Linear(hidden_dim, output_dim)

  def forward(self, x, h):
    a = self.ixh(x) + self.hxh(h)
    h = a.tanh()
    out = self.hxo(h).softmax()
    return out, h

'''
  Make RNN say "hello"
  Symbols : [h, e, l, o, end_token] -> dim = 5
'''

h = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
e = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
l = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
o = np.array([0.0, 0.0, 0.0, 1.0, 0.0])
end = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
dikt = ['h', 'e', 'l', 'o', 'done']

X = Tensor(np.array([h, e, l, l, o]))
y = Tensor(np.array([e, l, l, o, end]))

input_dim = 5
hidden_dim = 10
output_dim = input_dim

lr = 1e-2
epochs = 100

h0 = np.zeros((1, hidden_dim))
rnn = RNN(input_dim, hidden_dim, output_dim)
opt = optim.Adam(rnn.parameters(), learning_rate=lr)
loss_func = nn.NLL()


for e in range(epochs):
  h = Tensor(h0, requires_grad=False)
  if e == epochs - 1:
    print('h')
  for i in range(len(X.data)):
    out, h = rnn(X[i:i+1], h)
    if e == epochs - 1:
      print(dikt[np.argmax(out.data)])
      continue
      
    loss = loss_func(out, y[i:i+1])
    opt.zero_grad()
    loss.backward()
    opt.step()

