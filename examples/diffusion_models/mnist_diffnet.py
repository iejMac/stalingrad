import cv2
import numpy as np
from math import cos
from matplotlib import pyplot as plt

from stalingrad import nn
from stalingrad import optim
from stalingrad.tensor import Tensor
from stalingrad.utils import fetch_mnist, train_module, save_module, load_module

def play_sequence(sequence, frame_time=100):
  cv2.namedWindow("preview")
  for img in sequence:
    show_img = cv2.resize(img, (280, 280), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("img", show_img)
    if cv2.waitKey(frame_time) & 0xFF == ord('q'):
      break
  cv2.destroyWindow("preview")

def cos_alpha_schedule(t, T):
  s = 0.008
  ft = cos(((t/T + s)/(1+s)) * (np.pi / 2))**2
  f0 = cos((s/(1+s)) * (np.pi / 2))**2
  return ft/f0

def reg_beta_schedule(t, T):
  b0 = 1e-4
  bT = 0.02
  return b0 + (bT-b0) * (t/T)

def reg_alpha_bar(t, T):
  alpha_bar = 1.0
  for i in range(t):
    alpha_bar *= (1 - reg_beta_schedule(t, T))
  return alpha_bar

def forward_noising(X0s, t, T):
  alpha_bar_t = reg_alpha_bar(t, T)
  eps = np.random.normal(0.0, 1.0, size=X0s.shape)
  Xts = (alpha_bar_t**0.5)*X0s + ((1-alpha_bar_t)**0.5)*eps
  return Xts, eps

def get_dataset(X0, T):
  X_seq, Y_seq = [], []
  for t in range(T):
    Xt, eps = forward_noising(X0, t, T)
    X_seq.append(Xt)
    Y_seq.append(eps)

  X_seq, Y_seq = np.swapaxes(np.array(X_seq), 0, 1).reshape(-1, 1, 28, 28), np.swapaxes(np.array(Y_seq), 0, 1).reshape(-1, 1, 28, 28)

  p = np.random.permutation(len(X_seq))
  X_seq = X_seq[p]
  Y_seq = Y_seq[p]
  return X_seq, Y_seq
  
class DiffNet(nn.Module):
  def __init__(self, kern_size=3):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 16, kernel_size=kern_size, stride=1, padding="same")
    self.conv2 = nn.Conv2d(16, 16, kernel_size=kern_size, stride=1, padding="same")
    self.conv3 = nn.Conv2d(16, 1, kernel_size=kern_size, stride=1, padding="same")
  def forward(self, x):
    x = self.conv1(x).relu()
    x = self.conv2(x).relu()
    x = self.conv3(x)
    return x

X_train, _, X_test, _ = fetch_mnist(flatten=False, one_hot=True)
X_train, X_test = (np.expand_dims(X_train, 1)*2)/255.0 - 1.0, (np.expand_dims(X_test, 1)*2)/255.0 - 1.0 # normalize

T = 100
lr = 1e-2
X0 = X_train[:20]

X, Y = get_dataset(X0, T)
loss = nn.MSE()
# DN = DiffNet(kern_size=3)
DN = load_module("./examples/diffusion_models/diffnet.pkl")
opt = optim.Adam(DN.parameters(), lr)

# train_module(DN, opt, loss, X, Y, steps=500, batch_size=200)
# save_module(DN, "./examples/diffusion_models/diffnet.pkl")

xt = np.random.normal(0.0, 1.0, size=X[0:1].shape)
show_xt = xt - xt.min()
show_xt = ((show_xt / show_xt.max()) * 255.0).astype(np.uint8)
denoised_seq = [show_xt[0][0]]

xt = Tensor(xt)

for t in range(T):
  t_back = T-t
  eps = DN(xt)
  alpha_t = 1 - reg_beta_schedule(t_back, T)
  alpha_bar_t = reg_alpha_bar(t_back, T)

  xt_prev = (1/(alpha_t**0.5)) * (xt.data - ((1-alpha_t)/((1-alpha_bar_t)**0.5)) * eps.data)

  show_xt_prev = xt_prev - xt_prev.min()
  show_xt_prev = ((show_xt_prev / show_xt_prev.max()) * 255.0).astype(np.uint8) + 10
  denoised_seq.append(show_xt_prev[0][0])
  xt = Tensor(xt_prev)


play_sequence(denoised_seq, 10)
