import numpy as np
from stalingrad.tensor import Tensor

# -= Modules =-

class Module:
  def __init__(self):
    self.training = True
  def __call__(self, *x):
    return self.forward(*x)
  def parameters(self, parent="root"):
    params = {}
    for attr in self.__dict__:
      if isinstance(self.__dict__[attr], Tensor):
        params[parent + "." + attr] = self.__dict__[attr]
      elif isinstance(self.__dict__[attr], Module):
        params.update(self.__dict__[attr].parameters(parent + "." + attr))
    return params
  def training(self):
    self.training = True
  def eval(self):
    self.training = False
  
class Linear(Module):
  def __init__(self, in_neurons, out_neurons, use_bias=True):
    super().__init__()
    self.use_bias = use_bias
    self.shape = (in_neurons, out_neurons)

    self.weight = Tensor(np.random.uniform(-1., 1., size=self.shape)/np.sqrt(np.prod(self.shape)).astype(np.float32))
    self.bias = None
    if use_bias:
      self.bias = Tensor(np.random.uniform(-1., 1., (1, out_neurons)))

  def forward(self, x):
    result = x @ self.weight
    if self.use_bias:
      result += self.bias
    return result

class Conv2d(Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1), padding="valid", groups=1, use_bias=True):
    super().__init__()
    '''
      stride: kernel step size in y and x direction respectively
      padding (zero padding):
        int - pad width and height by padding
        tuple - pad height by padding[0] and width by padding[1]
        string :
          "valid" - no padding
          "same" - padding that keeps feature map same size
    '''
    self.groups = groups
    self.use_bias = use_bias
    self.shape = (out_channels, in_channels // groups, kernel_size, kernel_size)
    self.stride = (stride, stride) if isinstance(stride, int) else stride

    padding = (padding, padding) if isinstance(padding, int) else padding
    padding = [(0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])] if isinstance(padding, tuple) else padding
    self.padding = padding
  
    self.kernels = Tensor(np.random.uniform(-1., 1., size=self.shape)/np.sqrt(np.prod(self.shape)).astype(np.float32))
    self.bias = None

  def forward(self, x):
    if self.padding == "same":
      y_pad = (x.shape[-2] * self.stride[0] - x.shape[-2] - self.stride[0] + self.kernels.shape[-2] ) // 2
      x_pad = (x.shape[-1] * self.stride[1] - x.shape[-1] - self.stride[1] + self.kernels.shape[-1] ) // 2
      self.padding = [(0, 0), (0, 0), (y_pad, y_pad), (x_pad, x_pad)]
    if self.padding != "valid":
      x = x.pad(padding=self.padding)
    result = x.conv2d(self.kernels, stride=self.stride, groups=self.groups)
    if self.use_bias:
      if self.bias is None: # initialize bias after finding out output dim
        self.bias = Tensor(np.random.uniform(-1., 1., size=(1, *result.shape[1:])))
      result += self.bias
    return result

class ConvTranspose2d(Conv2d):
  def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1), padding="valid", groups=1, use_bias=True):
    super().__init__(in_channels, out_channels, kernel_size, stride, padding, groups, use_bias)
  def forward(self, x):
    if self.padding != "valid":
      x = x.pad(padding=self.padding)
    result = x.convtranspose2d(self.kernels, stride=self.stride, groups=self.groups)
    if self.use_bias:
      if self.bias is None:
        self.bias = Tensor(np.random.uniform(-1., 1., size=(1, *result.shape[1:])))
      result += self.bias
    return result

# -= Losses =-

class Loss(Module):
  def __init__(self, reduction=None, reduce_axis=0):
    super().__init__()
    self.reduction=reduction
    self.reduce_axis = reduce_axis
  def __call__(self, prediction, target):
    loss = self.forward(prediction, target)
    if self.reduction is not None:
      loss = getattr(loss, self.reduction)(axis=self.reduce_axis)
    return loss

class MSE(Loss):
  def __init__(self, reduction=None, reduce_axis=0):
    super().__init__(reduction, reduce_axis)
  def forward(self, prediction, target):
    return (prediction - target)**2

class NLL(Loss):
  def __init__(self, reduction=None, reduce_axis=0):
    super().__init__(reduction, reduce_axis)
  def forward(self, prediction, target):
    sum_axis = list(range(len(prediction.shape)))
    sum_axis.remove(self.reduce_axis)
    return (target * prediction.log() * (-1.0)).sum(axis=tuple(sum_axis))

class BCELoss(Loss):
  def __init__(self, reduction=None, reduce_axis=0):
    super().__init__(reduction, reduce_axis)
  def forward(self, prediction, target):
    sum_axis = list(range(len(prediction.shape)))
    sum_axis.remove(self.reduce_axis)
    return ((target * prediction.log() + (1.0 - target) * ((1.0 - prediction).log())) * (-1.0)).sum(axis=tuple(sum_axis))
