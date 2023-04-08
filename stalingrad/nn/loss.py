"""stalingrad loss functions"""

from stalingrad import Tensor
from stalingrad.nn import Module

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
