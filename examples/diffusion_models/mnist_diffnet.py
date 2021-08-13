import cv2
import numpy as np
from matplotlib import pyplot as plt

from stalingrad import nn
from stalingrad.utils import fetch_mnist

def play_sequence(sequence, frame_time=100):
  cv2.namedWindow("preview")
  for img in sequence:
    show_img = cv2.resize(img, (280, 280))
    cv2.imshow("img", show_img)
    if cv2.waitKey(frame_time) & 0xFF == ord('q'):
      break
  cv2.destroyWindow("preview")

class DiffNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 16, stride=1, padding="same")
    self.conv2 = nn.Conv2d(16, 16, stride=1, padding="same")
    self.conv3 = nn.Conv2d(16, 1, stride=1, padding="same")

X_train, _, X_test, _ = fetch_mnist(flatten=False, one_hot=True)
X_train, X_test = np.expand_dims(X_train, 1) / 255.0, np.expand_dims(X_test, 1) / 255.0

X_train, X_test = (X_train*255.0).astype(np.uint8), (X_test*255.0).astype(np.uint8)
seq = X_train[:20, 0]

play_sequence(seq, 1000)

