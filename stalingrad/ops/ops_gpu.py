import numpy as np
import pyopencl as cl

from stalingrad.tensor import Function

cl_ctx, cl_queue = None, None
def init_gpus():
  global cl_ctx, cl_queue
  if cl_queue is None:
    platform = cl.get_platforms()[0]
    devices = platform.get_devices()

    # TODO: for now lets do 1 GPU
    # devices = devices[:1]
    devices = [devices[5]]
    cl_ctx = cl.Context(devices)
    cl_queue = cl.CommandQueue(cl_ctx)
init_gpus()


class GPUBuffer:
  def fromCPU(data):
    return data
  def toCPU(data):
    return data


class ReLU(Function):
  def forward(func, x):
    func.save_tensors(x)
    return np.maximum(x, 0)
  def backward(func, passed_grad):
    x = func.saved_tensors[0]
    return passed_grad * (x >= 0)


