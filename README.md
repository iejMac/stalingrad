# stalingrad
our deep learning framework

# Setup:
Clone stalingrad into your home directory and run the setup script:
```
cd ~
git clone https://github.com/iejMac/stalingrad.git
cd stalingrad
python setup.py install
```

# Defining a model:
Define a model using stalingrad's nn API and pre-defined tensor operations. All you need to do is implement the forward pass and stalingrad will handle the backward pass for you.
```
from stalingrad import nn
class MyModel(nn.Module)
  def __init__(self, input_dim, hidden_dim, output_dim):
    super().__init__()
    self.linear1 = nn.Linear(input_dim, hidden_dim)
    self.linear2 = nn.Linear(hidden_dim, output_dim)
  def forward(self, x):
    x = self.linear1(x).relu()
    x = self.linear2(x).softmax()
    return x
```

# Training:
Simple training loop example:
```
from stalingrad import nn
from stalingrad import optim

epochs = 10
lr = 1e-2

X_train, Y_train = get_numpy_data()
X_train, Y_train = Tensor(X_train, requires_grad=False), Tensor(Y_train, requires_grad=False)

model = MyModel(784, 100, 10) # initialize model
optimizer = optim.SGD(model.parameters(), learning_rate=lr) # initialize optimizer with model parameters
loss_fn = nn.NLL(reduction="mean") # choose loss function

for e in epochs:
  output = model(X_train) # forward pass
  loss = loss_fn(output, Y_train) # calculate loss
  loss.backward() # pass loss backward to populate Tensor gradients
  optimizer.step() # apply Tensor gradients according to optimizer algorithm
  optimizer.zero_grad() # reset optimizer for next pass
```

###  Inspiration:
[https://github.com/geohot/tinygrad](https://github.com/geohot/tinygrad)

[https://github.com/karpathy/micrograd](https://github.com/karpathy/micrograd)
