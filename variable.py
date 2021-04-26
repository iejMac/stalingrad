class Variable:
  def __init__(self, value, name="", local_grads={}):
    self.value = value
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

def neg(x: Variable) -> Variable:
  return Variable(x.value * (-1), f"(-{x.name})", {
    x: -1
  })
def inv(x: Variable) -> Variable:
  return Variable(1 / x.value, f"(1/{x.name})", {
    x: -1 / ((x.value)**2)
  })
def add(x: Variable, y: Variable) -> Variable:
  return Variable(x.value + y.value, f"({x.name}+{y.name})", {
    x: 1,
    y: 1
  })
def mul(x: Variable, y: Variable) -> Variable:
  return Variable(x.value * y.value, f"({x.name}*{y.name})", {
    x: y.value,
    y: x.value
  })
