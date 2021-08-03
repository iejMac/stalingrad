import numpy as np
from stalingrad.tensor import Function

class ReLU(Function):
  def forward(func, x):
    func.save_tensors(x)
    return np.maximum(x, 0)
  def backward(func, passed_grad):
    x = func.saved_tensors[0]
    return passed_grad * (x >= 0)

class Log(Function):
  def forward(func, x):
    func.save_tensors(x)
    return np.log(x)
  def backward(func, passed_grad):
    x = func.saved_tensors[0]
    return passed_grad * 1/x

class Exp(Function):
  def forward(func, x):
    ret = np.exp(x)
    func.save_tensors(ret) # d(e^x)/dx = e^x
    return ret
  def backward(func, passed_grad):
    e_x = func.saved_tensors[0]
    return passed_grad * e_x

def unbroadcast(out, in_sh): # https://github.com/geohot/tinygrad/blob/master/tinygrad/ops_cpu.py (line 65)
  # It's possible to perform operations on tensors of different size f.e. Add((5, 5), (1, 5)) but for the backward
  # pass we need to remember to unbroadcast the output back to (1, 5)
  sum_axis = tuple([i for i in range(len(in_sh)) if in_sh[i]==1 and out.shape[i]>1]) if in_sh != (1,) else None
  return out.sum(axis=sum_axis).reshape(in_sh)

class Add(Function):
  def forward(func, x, y):
    func.save_tensors(x, y)
    return x+y
  def backward(func, passed_grad):
    x, y = func.saved_tensors
    return unbroadcast(passed_grad, x.shape), unbroadcast(passed_grad, y.shape)

class Sub(Function):
  def forward(func, x, y):
    func.save_tensors(x, y)
    return x-y
  def backward(func, passed_grad):
    x, y = func.saved_tensors
    return unbroadcast(passed_grad, x.shape), unbroadcast(-passed_grad, y.shape)

class Mul(Function):
  def forward(func, x, y):
    func.save_tensors(x, y)
    return x*y
  def backward(func, passed_grad):
    x, y = func.saved_tensors
    return unbroadcast(y*passed_grad, x.shape), unbroadcast(x*passed_grad, y.shape)

class Pow(Function):
  def forward(func, x, y):
    func.save_tensors(x, y)
    return x**y
  def backward(func, passed_grad):
    x, y = func.saved_tensors
    return unbroadcast(y * (x**(y-1.0)) * passed_grad, x.shape), unbroadcast((x**y) * np.log(x) * passed_grad, y.shape)

class Matmul(Function):
  def forward(func, x, y):
    func.save_tensors(x, y)
    return x @ y
  def backward(func, passed_grad):
    x, y = func.saved_tensors
    return passed_grad @ np.swapaxes(y, -2, -1), np.swapaxes(x, -2, -1) @ passed_grad

class Sum(Function):
  def forward(func, x, axis=None):
    ax = (axis,) if isinstance(axis, int) else axis
    func.save_tensors(x, ax)
    return x.sum(axis=ax) if ax is not None else np.array([x.sum()])
  def backward(func, passed_grad):
    x, axis = func.saved_tensors
    shape = [1 if (axis is None or ax in axis) else x.shape[ax] for ax in range(len(x.shape))]
    return passed_grad.reshape(shape) + np.zeros_like(x)

class Reshape(Function):
  def forward(func, x, shape=None):
    func.save_tensors(x, shape)
    return x.reshape(shape)
  def backward(func, passed_grad):
    x, _ = func.saved_tensors
    return passed_grad.reshape(x.shape)

class Slice(Function):
  def forward(func, x, inds=None):
    func.save_tensors(x.shape, inds)
    return x[inds]
  def backward(func, passed_grad):
    x_shape, inds = func.saved_tensors
    grad = np.zeros(x_shape)
    grad[inds] += passed_grad
    return grad

class Pad(Function):
  def forward(func, x, padding=None):
    padding = (padding, padding) if isinstance(padding, int) else padding
    padding = [padding for _ in range(len(x.shape))] if isinstance(padding, tuple) else padding
    func.save_tensors(padding)
    return np.pad(x, padding)
  def backward(func, passed_grad):
    padding, = func.saved_tensors
    return passed_grad[tuple(slice(s, passed_grad.shape[i] - e) for i, (s, e) in enumerate(padding))]

class Conv2d(Function):
  def forward(func, x, kernels, stride=(1, 1), groups=1):
    batch_size, in_ch, in_y, in_x = x.shape
    out_ch, k_in_ch, kern_y, kern_x = kernels.shape
    stride_y, stride_x = stride
    out_y, out_x = (in_y - kern_y + stride_y) // stride_y, (in_x - kern_x + stride_x) // stride_x

    assert k_in_ch * groups == in_ch
    assert out_ch % groups == 0 and in_ch % groups == 0
    g_out_ch = out_ch // groups

    grouped_x = x.reshape(batch_size, groups, k_in_ch, in_y, in_x)
    strided_x = np.lib.stride_tricks.as_strided(grouped_x,
                                                shape=(batch_size, groups, k_in_ch, out_y, out_x, kern_y, kern_x),
                                                strides=(*grouped_x.strides[0:3], grouped_x.strides[3]*stride_y, grouped_x.strides[4]*stride_x, *grouped_x.strides[3:5]),
                                                writeable=False)
    
    grouped_kernels = kernels.reshape(groups, g_out_ch, k_in_ch, kern_y, kern_x)

    func.save_tensors(strided_x, grouped_kernels, x.shape, stride)
    output = np.zeros((batch_size, groups, out_y, out_x, g_out_ch))
    for group in range(groups):
      output[:, group] += np.tensordot(strided_x[:, group], grouped_kernels[group], ((1, 4, 5), (1, 2, 3)))

    return np.moveaxis(output, 4, 2).reshape(batch_size, out_ch, out_y, out_x)

  def backward(func, passed_grad):
    batch_size, out_ch, out_y, out_x = passed_grad.shape
    strided_x, grouped_kernels, x_shape, stride = func.saved_tensors
    groups, g_out_ch, k_in_ch, kern_y, kern_x = grouped_kernels.shape
    _, in_ch, in_y, in_x = x_shape
    stride_y, stride_x = stride

    d_kernels = np.zeros_like(grouped_kernels)
    d_input = np.zeros((batch_size, groups, k_in_ch, in_y, in_x))
    grouped_passed_grad = passed_grad.reshape(batch_size, groups, g_out_ch, out_y, out_x)

    for group in range(groups):
      d_kernels[group] += np.tensordot(grouped_passed_grad[:, group], strided_x[:, group], ((0, 2, 3), (0, 2, 3)))

    for r in range(out_y):
      for c in range(out_x):
        for group in range(groups):
          r0, c0 = r * stride_y, c * stride_x
          d_input[:, group, :, r0:r0+kern_y, c0:c0+kern_x] += np.tensordot(grouped_passed_grad[:, group, :, r, c], grouped_kernels[group], ((1,), (0,)))
    
    return d_input.reshape(batch_size, groups*k_in_ch, in_y, in_x), d_kernels.reshape(groups*g_out_ch, k_in_ch, kern_y, kern_x)

class ConvTranspose2d(Function):
  def forward(func, x, kernels, stride=(1, 1), groups=1):
    batch_size, in_ch, in_y, in_x = x.shape
    out_ch, k_in_ch, kern_y, kern_x = kernels.shape
    stride_y, stride_x = stride
    out_y, out_x = (in_y - 1) * stride_y + kern_y, (in_x - 1) * stride_x + kern_x

    assert k_in_ch * groups == in_ch
    assert out_ch % groups == 0 and in_ch % groups == 0
    g_out_ch = out_ch // groups

    output = np.zeros((batch_size, groups, g_out_ch, out_y, out_x))
    grouped_x = x.reshape(batch_size, groups, k_in_ch, in_y, in_x)
    grouped_kernels = kernels.reshape(groups, g_out_ch, k_in_ch, kern_y, kern_x)

    func.save_tensors(grouped_x, grouped_kernels, x.shape, stride)

    for r in range(in_y):
      for c in range(in_x):
        for group in range(groups):
          r0, c0 = r * stride_y, c * stride_x
          output[:, group, :, r0:r0+kern_y, c0:c0+kern_x] += np.tensordot(grouped_x[:, group, :, r, c], grouped_kernels[group], ((1,), (1,)))

    return output.reshape(batch_size, groups*g_out_ch, out_y, out_x)
  def backward(func, passed_grad):
    batch_size, out_ch, out_y, out_x = passed_grad.shape
    grouped_x, grouped_kernels, x_shape, stride = func.saved_tensors
    groups, g_out_ch, k_in_ch, kern_y, kern_x = grouped_kernels.shape
    _, in_ch, in_y, in_x = x_shape
    stride_y, stride_x = stride

    d_kernels = np.zeros((groups, k_in_ch, g_out_ch, kern_y, kern_x))
    d_input = np.zeros((batch_size, groups, in_y, in_x, k_in_ch))
    grouped_passed_grad = passed_grad.reshape(batch_size, groups, g_out_ch, out_y, out_x)
    strided_passed_grad = np.lib.stride_tricks.as_strided(grouped_passed_grad,
                                                shape=(batch_size, groups, g_out_ch, in_y, in_x, kern_y, kern_x),
                                                strides=(*grouped_passed_grad.strides[0:3], grouped_passed_grad.strides[3]*stride_y, grouped_passed_grad.strides[4]*stride_x, *grouped_passed_grad.strides[3:5]),
                                                writeable=False)

    for group in range(groups):
      d_input[:, group] += np.tensordot(strided_passed_grad[:, group], grouped_kernels[group], ((1, 4, 5), (0, 2, 3)))
      d_kernels[group] += np.tensordot(grouped_x[:, group], strided_passed_grad[:, group], ((0, 2, 3), (0, 2, 3)))

    return np.moveaxis(d_input, 4, 2).reshape(batch_size, in_ch, in_y, in_x), np.moveaxis(d_kernels, 1, 0).reshape(groups*g_out_ch, k_in_ch, kern_y, kern_x)
