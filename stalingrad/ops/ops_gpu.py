import functools
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

def empty_buf(shape, dtype=np.float32, zero=False):
  data = np.empty(shape, dtype=dtype) if not zero else np.zeros(shape, dtype=dtype)
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
  prg.backward_unary_op(cl_queue, [np.prod(x.shape)], None, x.buf, upstream_grad.buf, result_grad.buf)
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

def binary_broadcast(x_shape, y_shape):
  n_dims = max(len(x_shape), len(y_shape))
  shape_x, shape_y = np.ones(n_dims, dtype=np.int32), np.ones(n_dims, dtype=np.int32)
  shape_x[:len(x_shape)] = np.array(x_shape, dtype=np.int32)
  shape_y[:len(y_shape)] = np.array(y_shape, dtype=np.int32)
  if not np.all((shape_x == 1) | (shape_y == 1) | (shape_x == shape_y)):
    raise Exception(f"binary op unbroadcastable shape mismatch: {x_shape} vs {y_shape}")
  shape_ret = np.maximum(shape_x, shape_y)

  dimlist, complist = [], [] # note: len(dimlist) may be less than n_dims
  def push(dim, comp):
    if len(complist) > 0 and complist[-1] == comp:
      dimlist[-1] *= dim
    elif comp != (False, False):
      dimlist.append(dim); complist.append(comp)
  for i in range(n_dims): # group together any adjacent dimensions that we can to simplify broadcasting
    push(np.int32(max(shape_x[i], shape_y[i])), (shape_x[i] > 1, shape_y[i] > 1))

  return shape_ret, dimlist, complist

@functools.lru_cache
def get_binop_prg(cl_ctx, code, complist):
  ndims = len(complist)
  args = "".join([f", int d{i}" for i in range(ndims)] + [f", int p{i}" for i in range(ndims-1)])
  compute_idx_rets = "".join([f"\n    int idx_ret{i} = (gid0 / {f'p{i}' if i < ndims-1 else '1'}) % d{i};" for i in range(ndims)])

  idx_exprs = ["0", "0"] # [idx_x, idx_y]
  for i in range(ndims):
    for j in range(2):
      if complist[i][j]:
        idx_exprs[j] = "idx_ret%d + d%d*(%s)" % (i, i, idx_exprs[j])

  return cl.Program(cl_ctx, """__kernel void binop(__global const float *x_g, __global const float *y_g, __global float *res_g"""+args+""") {
    int gid0 = get_global_id(0);"""+compute_idx_rets+"""
    float a = x_g["""+idx_exprs[0]+"""];
    float b = y_g["""+idx_exprs[1]+"""];
    res_g[gid0] = """+code+""";\n}""").build()

def binary_op(code, x, y):
  shape_ret, dimlist, complist = binary_broadcast(x.shape, y.shape)
  prod_list = np.array(dimlist, dtype=np.int32)[-1::-1].cumprod(dtype=np.int32)[-1::-1] # take cumprod from back to front

  prg = get_binop_prg(cl_ctx, code, tuple(complist))
  ret = empty_buf(shape_ret, zero=True)
  prg.binop(cl_queue, [prod_list[0]] if len(dimlist) > 0 else [1], None, x.buf, y.buf, ret.buf, *dimlist, *(prod_list[1:]))
  return ret

def unbroadcast(func, out, in_sh):
  sum_axis = [i for i in range(len(in_sh)) if in_sh[i]==1 and out.shape[i]>1] if in_sh != (1,) else None
  return reduce_op("out += a", "out", out, sum_axis)

class Add(Function):
  def forward(func, x, y):
    func.save_tensors(x.shape, y.shape)
    return binary_op('a+b', x, y)
  def backward(func, grad_output):
    grad_x, grad_y = grad_output, grad_output
    shape_x, shape_y = func.saved_tensors
    return unbroadcast(func, grad_x, shape_x), unbroadcast(func, grad_y, shape_y),

class Sub(Function):
  def forward(func, x, y):
    func.save_tensors(x.shape, y.shape)
    return binary_op('a-b', x, y)
  def backward(func, grad_output):
    grad_x, grad_y = grad_output, unary_op('-x', grad_output)
    shape_x, shape_y = func.saved_tensors
    return unbroadcast(func, grad_x, shape_x), unbroadcast(func, grad_y, shape_y),

class Mul(Function):
  def forward(func, x, y):
    func.save_tensors(x, y)
    return binary_op('a*b', x, y)

  def backward(func, grad_output):
    x,y = func.saved_tensors
    grad_x = binary_op('a*b', y, grad_output)
    grad_y = binary_op('a*b', x, grad_output)
    return unbroadcast(func, grad_x, x.shape), unbroadcast(func, grad_y, y.shape),

class Pow(Function):
  def forward(func, x, y):
    func.save_tensors(x, y)
    return binary_op('pow(a,b)', x, y)

  def backward(func, grad_output):
    x,y = func.saved_tensors
    grad_x = binary_op('a*b', grad_output,
                      binary_op('b * (pow((float)a, (float)(b-1.0)))', x, y))
    grad_y = binary_op('a*b', grad_output,
                      binary_op('pow(a, (float)b) * log(a);', x, y))
    return unbroadcast(func, grad_x, x.shape), unbroadcast(func, grad_y, y.shape),


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
    return transpose_op(passed_grad, order)

# TODO: IMPLEMENT PROPERLY (THIS IS JUST CPU IMPLEM)
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


# REDUCE OPS #

def reduce_op(code, code2, inp, axis=None, start="0.0"):
  if axis is None:
    # full reduce
    osize = [1]*len(inp.shape)
  else:
    osize = np.array(inp.shape)
    osize[list(axis)] = 1
  ret = empty_buf(osize)
  if axis is None:
    ret.shape = (1,)

  # TODO: this is insanely slow
  reduce_op_kernel = """
    __kernel void reduce_op(__global const float *a_g, int sz, __global float *res_g, int prod, int n_dims,
                         __global const int *shape_x, __global const int *shape_ret) {
      int gid = get_global_id(0);
      float out = """+start+""";
      for (int x = 0; x < sz; x++) {
        int idx = 0;  // compute index into a_g
        int tprod = prod;
        int tsz = sz;
        for (int dim = 0; dim < n_dims; dim++) {
          idx *= shape_x[dim];
          if (shape_x[dim] == shape_ret[dim]) {   // dim from gid, don't reduce
            tprod /= shape_x[dim];
            idx += (gid / tprod) % shape_x[dim];
          } else {  // dim from x
            tsz /= shape_x[dim];
            idx += (x / tsz) % shape_x[dim];
          }
        }
        float a = a_g[idx];
        """+code+""";
      }
      res_g[gid] = """+code2+""";
    }
  """
  prg = cl.Program(cl_ctx, reduce_op_kernel).build()

  prg.reduce_op(cl_queue, [np.prod(osize)], None, inp.buf,
    np.int32(np.prod(inp.shape)//np.prod(osize)), ret.buf,
    np.int32(np.prod(osize)), np.int32(len(osize)),
    GPUBuffer(np.array(inp.shape, dtype=np.int32)).buf,
    GPUBuffer(np.array(osize, dtype=np.int32)).buf
  )
  return ret

class Sum(Function):
  def forward(func, input, axis=None):
    if isinstance(axis, int): axis = [axis]
    func.save_tensors(input, axis)
    ret = reduce_op("out += a", "out", input, axis=axis)
    if axis is not None:
      ret.shape = tuple([input.shape[i] for i in range(len(input.shape)) if i not in axis])
    return ret

  def backward(func, passed_grad):
    input, axis = func.saved_tensors
    shape = [1 if axis is None or i in axis else input.shape[i] for i in range(len(input.shape))]
    output = GPUBuffer(passed_grad)
    return binary_op('a+b', output, empty_buf(input.shape, zero=True))


# MATMUL #

class Matmul(Function):
  def forward(func, input, weight):
    assert input.shape[-1] == weight.shape[-2]
    cnt = np.prod(input.shape[0:-2]) if len(input.shape) > 2 else 1
    isize, msize, osize = np.int32(input.shape[-2]), np.int32(input.shape[-1]), np.int32(weight.shape[-1])
    ret = empty_buf(list(input.shape[0:-2])+[isize, osize])

    matmul_kernel = """
      __kernel void matmul(
        __global const float *input, __global const float *weight, __global float *res,
        int isize, int is0, int is1, int msize, int ws0, int ws1, int osize
     ) {
        int stride = get_global_id(2);
        int X = get_global_id(0); // isize
        int Y = get_global_id(1); // osize
        float ret = 0.0;
        for (int x = 0; x < msize; x++) {
          ret += input[X * is0 + x * is1 + isize*msize*stride] *
            weight[Y * ws0 + x * ws1 + msize*osize*stride];
        }
        res[X * osize + Y + isize*osize*stride] = ret;
      }
    """
    prg = cl.Program(cl_ctx, matmul_kernel).build()
    matmul = prg.__getattr__("matmul")
    func.save_tensors(input, weight, matmul, cnt)

    # (isize,msize) x (msize,osize) = (isize,osize)
    matmul(cl_queue, [isize, osize, cnt], None,
      input.buf, weight.buf, ret.buf, isize,
      msize, np.int32(1), msize, np.int32(1), osize, osize)
    return ret

  def backward(func, grad_output):
    input, weight, matmul, cnt = func.saved_tensors
    isize, msize, osize = np.int32(input.shape[-2]), np.int32(input.shape[-1]), np.int32(weight.shape[-1])

    grad_input = empty_buf(input.shape)
    grad_weight = empty_buf(weight.shape)

    # (isize,osize) x (msize,osize) = (isize,msize)
    matmul(cl_queue, [isize, msize, cnt], None,
      grad_output.buf, weight.buf, grad_input.buf, isize,
      osize, np.int32(1), osize, osize, np.int32(1), msize)

    # (isize,msize) x (isize,osize) = (msize,osize)
    matmul(cl_queue, [msize, osize, cnt], None,
      input.buf, grad_output.buf, grad_weight.buf, msize,
      np.int32(1), msize, isize, np.int32(1), osize, osize)

    return grad_input, grad_weight


