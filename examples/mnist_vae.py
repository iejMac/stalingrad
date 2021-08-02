import numpy as np

from stalingrad import nn
from stalingrad.tensor import Tensor

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
    x = self.encoder(x)

    # Samp:
    x = Tensor(x.data[:, :self.latent_dim])

    x = self.decoder(x)
    return x

  
test = Tensor(np.ones((10, 28, 28)), requires_grad=False)
vae = MnistVAE(20)
out = vae(test)
print(out.shape)
