import numpy as np

from stalingrad import nn
from stalingrad import optim
from stalingrad.tensor import Tensor
from stalingrad.utils import train_module, fetch_mnist

class Encoder(nn.Module):
  def __init__(self, latent_dim):
    self.conv1 = nn.Conv2d(1, 8, 4, stride=2)
    self.conv2 = nn.Conv2d(8, 4, 4, stride=2)
    self.lin = nn.Linear(100, 2 * latent_dim)
  def forward(self, x):
    x = self.conv1(x).relu()
    x = self.conv2(x).relu()
    x = x.reshape(shape=(x.shape[0], -1))
    x = self.lin(x)
    return x

class Decoder(nn.Module):
  def __init__(self, latent_dim):
    self.lin = nn.Linear(latent_dim, 100)
    self.conv1T = nn.ConvTranspose2d(4, 8, 4, stride=2)
    self.conv2T = nn.ConvTranspose2d(8, 8, 4, stride=2)
    self.conv3T = nn.ConvTranspose2d(8, 1, 3, stride=1)
  def forward(self, x):
    x = self.lin(x)
    x = x.reshape(shape=(-1, 4, 5, 5))
    x = self.conv1T(x).relu()
    x = self.conv2T(x).relu()
    x = self.conv3T(x).sigmoid()
    return x

class MnistVAE(nn.Module):
  def __init__(self, latent_dim):
    self.latent_dim = latent_dim
    self.encoder = Encoder(latent_dim)
    self.decoder = Decoder(latent_dim)
  def forward(self, x):
    x = x.reshape(shape=(-1, 1, 28, 28))
    lat = self.encoder(x)
    mu, logvar = lat[:, :self.latent_dim], lat[:, self.latent_dim:]

    # reparameterization:
    if self.training:
      epsilon = Tensor(np.random.normal(0.0, 1.0, size=(x.shape[0], self.latent_dim)), requires_grad=False)
      samp = mu + epsilon * ((logvar * 0.5).exp())
    else:
      samp = mu

    out = self.decoder(samp)
    return out, mu, logvar

def VAELoss(preds, targets):
  x, mu, logvar = preds
  kl_loss = ((logvar.exp() - logvar - 1 + mu**2) * 0.5).sum(axis=1).mean(axis=0)
  reconstruction_loss = nn.NLL(reduction="mean")(x, targets)
  return reconstruction_loss + kl_loss
  
X_train, _, X_test, _ = fetch_mnist(flatten=False, one_hot=True)
X_train, X_test = np.expand_dims(X_train, 1) / 255.0, np.expand_dims(X_test, 1) / 255.0

if __name__ == "__main__":
  steps = 100
  batch_size = 200
  latent_dim = 20
  lr = 1e-3

  mod = MnistVAE(latent_dim)
  optimizer = optim.Adam(mod.parameters(), learning_rate=lr)
  loss_func = VAELoss
  train_module(mod, optimizer, loss_func, X_train, X_train, steps, batch_size)
