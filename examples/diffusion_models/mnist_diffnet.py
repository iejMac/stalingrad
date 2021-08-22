import cv2
import numpy as np
from math import cos
from matplotlib import pyplot as plt

from stalingrad import nn
from stalingrad.utils import fetch_mnist

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

def forward_noising(X0s, t, T):
  alpha_t = cos_alpha_schedule(t, T)
  eps = np.random.normal(0.0, 1.0, size=X0s.shape)
  Xts = (alpha_t**0.5)*X0s + ((1-alpha_t)**0.5)*eps
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
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 16, stride=1, padding="same")
    self.conv2 = nn.Conv2d(16, 16, stride=1, padding="same")
    self.conv3 = nn.Conv2d(16, 1, stride=1, padding="same")

X_train, _, X_test, _ = fetch_mnist(flatten=False, one_hot=True)
X_train, X_test = (np.expand_dims(X_train, 1)*2)/255.0 - 1.0, (np.expand_dims(X_test, 1)*2)/255.0 - 1.0 # normalize

T = 100
X0 = X_train[:20]

X, Y = get_dataset(X0, T)

test_noised = X[:500, 0]

seq = []
for img in test_noised:
  show_img = img - img.min()
  show_img = ((show_img / show_img.max()) * 255.0).astype(np.uint8)
  seq.append(show_img)

play_sequence(seq, 50)






