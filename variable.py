import numpy as np

class Variable:
  def __init__(self, value, name="", local_grads={}):
    self.value = float(value)
    self.local_grads = local_grads
    self.name = name
  def __repr__(self):
    return f'Variable "{self.name}"'
  def __neg__(self):
    return neg(self)
  def __inv__(self):
    return inv(self)
  def __add__(self, x):
    return add(self, x)
  def __sub__(self, x):
    return add(self, neg(x))
  def __mul__(self, x):
    return mul(self, x)
  def __truediv__(self, x):
    return mul(self, inv(x))
  def __pow__(self, x):
    return pow(self, x)

def handle_number(func):
  def inner(x, y):
    if not isinstance(y, Variable):
      return func(x, Variable(float(y), "constant"))
    return func(x, y)
  return inner

def neg(x: Variable) -> Variable:
  return Variable(x.value * (-1), f"(-{x.name})", {
    x: -1
  })
def inv(x: Variable) -> Variable:
  return Variable(1 / x.value, f"(1/{x.name})", {
    x: -1 / ((x.value)**2)
  })
@handle_number
def add(x: Variable, y: Variable) -> Variable:
  return Variable(x.value + y.value, f"({x.name}+{y.name})", {
    x: 1,
    y: 1
  })
@handle_number
def mul(x: Variable, y: Variable) -> Variable:
  return Variable(x.value * y.value, f"({x.name}*{y.name})", {
    x: y.value,
    y: x.value
  })
@handle_number
def pow(x: Variable, y: Variable) -> Variable:
  print(x.value)
  return Variable(x.value ** y.value, f"({x.name}^{y.name})", {
    x: y.value * (x.value ** (y.value - 1.0)),
    y: np.log(x.value) * (x.value ** y.value)
  })
