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
    # TODO: in the future for more devices we can maintain a context and queue for each
    # and pass index info to GPUBuffer
    devices = [devices[0]]
    cl_ctx = cl.Context(devices)
    cl_queue = cl.CommandQueue(cl_ctx)
init_gpus()


class GPUBuffer:
  def __init__(self, hostbuf):
    self.shape, self.dtype = hostbuf.shape, hostbuf.dtype

    mf = cl.mem_flags
    # GPUBuffer is flat, ops should know how to handle this based on shape info
    self.buf = hostbuf.buf if isinstance (hostbuf, GPUBuffer) else \
      cl.Buffer(cl_ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=hostbuf.ravel(), size=0)
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

# UNARY OPS #

def unary_op(code, x):
  result = empty_buf(x.shape, x.dtype)
  unary_op_kernel = """
    __kernel void unary_op(__global const float *input, __global float *output) {
      int gid = get_global_id(0);
      float x = input[gid];
      output[gid] = """+code+""";
    }
  """
  prg = cl.Program(cl_ctx, unary_op_kernel).build()
  prg.unary_op(cl_queue, [np.prod(x.shape)], None, x.buf, result.buf)
  return result
def backward_unary_op(code, x, upstream_grad):
  result_grad = empty_buf(x.shape, x.dtype)
  backward_unary_op_kernel = """
    __kernel void backward_unary_op(__global const float *input, __global const float *upstream_gradient, __global float *output) {
      int gid = get_global_id(0);
      float x = input[gid];
      float up_grad = upstream_gradient[gid];
      output[gid] = """+code+""";
    }
  """
  prg = cl.Program(cl_ctx, backward_unary_op_kernel).build()
  prg.backward_unary_op(cl_queue, [np.prod(x.shape)], None, x.buf, passed_grad.buf, result_grad.buf)
  return result_grad


# TODO: THIS SHIT IS SO SLOW
class ReLU(Function):
  def forward(func, x):
    func.save_tensors(x)
    return unary_op("max(0.0f, x)", x)

  def backward(func, passed_grad): # upstream_gradient[gid] * (input[gid] > 0.0f)
    x = func.saved_tensors[0]
    return backward_unary_op("up_grad * (x > 0.0f)", x, passed_grad)

class Log(Function):
  def forward(func, x):
    func.save_tensors(x)
    return unary_op("log(x)", x)
  def backward(func, passed_grad):
    x = func.saved_tensors[0]
    return backward_unary_op("up_grad * (1 / x)", x, passed_grad)

class Exp(Function):
  def forward(func, x):
    ret = unary_op("exp(x)", x)
    func.save_tensors(ret) # d(e^x)/dx = e^x
    return ret
  def backward(func, passed_grad):
    e_x = func.saved_tensors[0]
    return backward_unary_op("up_grad * x", e_x, passed_grad)

# BINARY OPS #

def binary_op(code, x, y):
  result = empty_buf(x.shape, x.dtype)
  unary_op_kernel = """
    __kernel void unary_op(__global const float *input, __global float *output) {
      int gid = get_global_id(0);
      float x = input[gid];
      output[gid] = """+code+""";
    }
  """
  prg = cl.Program(cl_ctx, unary_op_kernel).build()
  prg.unary_op(cl_queue, [np.prod(x.shape)], None, x.buf, result.buf)
  return result

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

# SHAPE OPS #

class Reshape(Function):
  def forward(func, x, shape=None):
    assert np.prod(x.shape) == np.prod(shape) # target tensor needs to have same n_elements
    func.save_tensors(x.shape) # save old shape
    ret = GPUBuffer(x)
    ret.shape = shape
    return ret
  def backward(func, passed_grad):
    shape = func.saved_tensors[0]
    ret_grad = GPUBuffer(passed_grad)
    ret_grad.shape = shape
    return ret_grad


def transpose_op(x, order):
  res_shape = np.array(x.shape)[np.array(order)]
  result = empty_buf(res_shape, x.dtype)
  shape_buf = GPUBuffer(np.array(x.shape).astype(np.int32))
  order_buf = GPUBuffer(np.array(order).astype(np.int32))

  transpose_op_kernel = """
		__kernel void transpose_op(__global const float *input, __global float *output,
															 __global const int *order, __global const int *shape,
															 int order_len) {
				int gid = get_global_id(0);
				int gi = gid;
				int output_index = 0;

				for (int i = order_len - 1; i >= 0; i--) {
						int stride = 1;
						for (int j = order[i] + 1; j < order_len; j++) {
								stride *= shape[j];
						}
						output_index += (gi % shape[order[i]]) * stride;
						gi /= shape[order[i]];
				}

				output[gid] = input[output_index];
		}
  """
  prg = cl.Program(cl_ctx, transpose_op_kernel).build()
  prg.transpose_op(cl_queue, [np.prod(x.shape)], None,
    x.buf, result.buf,
    order_buf.buf, shape_buf.buf,
    np.int32(len(order)),
  )
  return result 

class Transpose(Function):
  def forward(func, x, order=(1, 0)):
    func.save_tensors(order)
    return transpose_op(x, order)
  def backward(func, passed_grad):
    order, = func.saved_tensors
    return transpose_op(x, order)


# TODO: IMPLEMENT PROPERLY (THIS IS JUST CPU IMPLEM
class Slice(Function):
  def forward(func, x, inds=None):
    func.save_tensors(x.shape, inds)
    x = x.toCPU()
    out_buf = x[inds]
    return GPUBuffer(out_buf)
  def backward(func, passed_grad):
    x_shape, inds = func.saved_tensors
    passed_grad = passed_grad.toCPU()
    grad = np.zeros(x_shape)
    grad[inds] += passed_grad
    return GPUBuffer(grad)
