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
  def __init__(self, hostbuf):
    self.shape, self.dtype = hostbuf.shape, hostbuf.dtype

    mf = cl.mem_flags
    self.buf = cl.Buffer(cl_ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=hostbuf, size=0)
  def __repr__(self):
    return f"GPUBuffer of shape {self.shape}"
  def fromCPU(data):
    return GPUBuffer(hostbuf=data)
  def toCPU(self):
    result = np.empty(self.shape, dtype=self.dtype)
    cl.enqueue_copy(cl_queue, result, self.buf)
    return result

def empty_buf(shape, dtype=np.float32):
  data = np.empty(shape, dtype=dtype)
  buf = GPUBuffer(data)
  return buf


class ReLU(Function):
  def forward(func, x):
    func.save_tensors(x)
    result = empty_buf(x.shape, x.dtype)

    kernel_code = """
    __kernel void relu(__global const float *input, __global float *output) {
        int gid = get_global_id(0);
        output[gid] = max(0.0f, input[gid]);
    }
    """
    prg = cl.Program(cl_ctx, kernel_code).build()
    prg.relu(cl_queue, x.shape, None, x.buf, result.buf)
    return result
  def backward(func, passed_grad):
    x = func.saved_tensors[0]
    return passed_grad * (x >= 0)


