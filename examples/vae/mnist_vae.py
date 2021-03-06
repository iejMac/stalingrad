import cv2
import numpy as np
from matplotlib import pyplot as plt

from stalingrad import nn
from stalingrad import optim
from stalingrad.tensor import Tensor
from stalingrad.utils import train_module, fetch_mnist, save_module, load_module

def play_sequence(sequence, frame_time=100):
  cv2.namedWindow("preview")
  for img in sequence:
    show_img = cv2.resize(img, (280, 280), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("img", show_img)
    if cv2.waitKey(frame_time) & 0xFF == ord('q'):
      break
  cv2.destroyWindow("preview")

class Encoder(nn.Module):
  def __init__(self, latent_dim):
    self.conv1 = nn.Conv2d(1, 96, 4, stride=1)
    self.conv2 = nn.Conv2d(96, 96, 4, stride=1)
    self.conv3 = nn.Conv2d(96, 96, 4, stride=1)
    self.conv4 = nn.Conv2d(96, 96, 3, stride=1)
    self.conv5 = nn.Conv2d(96, 1, 3, stride=1)
    self.lin = nn.Linear(225, 2 * latent_dim)
  def forward(self, x):
    x = self.conv1(x).relu()
    x = self.conv2(x).relu()
    x = self.conv3(x).relu()
    x = self.conv4(x).relu()
    x = self.conv5(x).relu()
    x = x.reshape(shape=(x.shape[0], -1))
    x = self.lin(x)
    return x

class EncoderLin(nn.Module):
  def __init__(self, latent_dim):
    self.lin1 = nn.Linear(784, latent_dim ** 2)
    self.lin2 = nn.Linear(latent_dim ** 2, 2 * latent_dim)
  def forward(self, x):
    x = self.lin1(x).relu()
    x = self.lin2(x)
    return x

class Decoder(nn.Module):
  def __init__(self, latent_dim):
    self.lin = nn.Linear(latent_dim, 324)
    self.conv1T = nn.ConvTranspose2d(4, 96, 3, stride=2)
    self.conv2T = nn.ConvTranspose2d(96, 128, 3, stride=1)
    self.conv3T = nn.ConvTranspose2d(128, 128, 3, stride=1)
    self.conv4T = nn.ConvTranspose2d(128, 128, 3, stride=1)
    self.conv5T = nn.ConvTranspose2d(128, 1, 4, stride=1)
  def forward(self, x):
    x = self.lin(x)
    x = x.reshape(shape=(-1, 4, 9, 9))
    x = self.conv1T(x).relu()
    x = self.conv2T(x).relu()
    x = self.conv3T(x).relu()
    x = self.conv4T(x).relu()
    x = self.conv5T(x).sigmoid()
    return x

class DecoderLin(nn.Module):
  def __init__(self, latent_dim):
    self.lin1 = nn.Linear(latent_dim, latent_dim ** 2)
    self.lin2 = nn.Linear(latent_dim ** 2, 784)
  def forward(self, x):
    x = self.lin1(x).relu()
    x = self.lin2(x).sigmoid()
    return x

class MnistVAE(nn.Module):
  def __init__(self, latent_dim):
    self.latent_dim = latent_dim
    self.encoder = EncoderLin(latent_dim)
    self.decoder = DecoderLin(latent_dim)
  def forward(self, x):
    # x = x.reshape(shape=(-1, 1, 28, 28))
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
  kl_loss = ((logvar.exp() - logvar - 1 + mu**2) * 0.5).sum(axis=(0, 1))
  reconstruction_loss = nn.BCELoss(reduction="sum")(x, targets)
  print(kl_loss, reconstruction_loss)
  return reconstruction_loss + kl_loss
  
# X_train, _, X_test, _ = fetch_mnist(flatten=False, one_hot=True)
X_train, _, X_test, y_test = fetch_mnist(flatten=True, one_hot=False)
# X_train, X_test = np.expand_dims(X_train, 1) / 255.0, np.expand_dims(X_test, 1) / 255.0 
X_train, X_test = X_train / 255.0, X_test / 255.0 
train = False

if __name__ == "__main__":
  if train:
    lr = 1e-3
    steps = 1500
    latent_dim = 20
    batch_size = 256

    mod = MnistVAE(latent_dim)
    optimizer = optim.Adam(mod.parameters(), learning_rate=lr)
    loss_func = VAELoss
    train_module(mod, optimizer, loss_func, X_train, X_train, steps, batch_size)
    save_module(mod, "examples/mnist_vae.pkl")
  else:
    mod = load_module("examples/mnist_vae.pkl")
    mod.eval()

    '''
    for i, y in enumerate(y_test):
      if y == 2:
        print(i)
    quit()
    '''

    one = Tensor(X_test[31:32])
    # two = Tensor(X_test[1:2])
    two = Tensor(X_test[512:513])
    four = Tensor(X_test[95:96])
    five = Tensor(X_test[23:24])
    eight = Tensor(X_test[128:129])
    nine = Tensor(X_test[9:10])

    _, one_l, _ = mod(one)
    _, two_l, _ = mod(two)
    _, four_l, _ = mod(four)
    _, five_l, _ = mod(five)
    _, eight_l, _ = mod(eight)
    _, nine_l, _ = mod(nine)

    step_count = 6
    one_l = one_l.data
    two_l = two_l.data
    four_l = four_l.data
    five_l = five_l.data
    eight_l = eight_l.data
    nine_l = nine_l.data

    top_row = [five_l * (1 - i/step_count) + eight_l * (i/step_count) for i in range(step_count)]
    # bot_row = [one_l * (1 - i/step_count) + nine_l * (i/step_count) for i in range(step_count)]
    bot_row = [one_l * (1 - i/step_count) + nine_l * (i/step_count) for i in range(step_count)]

    # sequence = []
    output = np.zeros((6 * 28, 6 * 28))

    for i in range(step_count):
      for j in range(step_count):
        mix_l = top_row[i] * (1 - j/step_count) + bot_row[i] * (j/step_count)
        out = mod.decoder(Tensor(mix_l))
        img = (out[0].data * 255).astype(np.uint8).reshape(28, 28)
        # sequence.append(img)
        output[j*28:(j+1)*28, i*28:(i+1)*28] = img

    # play_sequence(sequence, 2000)
    plt.imshow(output, cmap="gray")
    plt.show()

    '''
    gaus = Tensor(np.random.normal(0.0, 1.0, size=(1, mod.latent_dim)))
    # img = (mod.decoder(gaus)[0, 0].data * 255).astype(int)
    img = (mod.decoder(gaus)[0].data * 255).astype(int).reshape(28, 28)
    '''
