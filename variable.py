
class Variable:
  def __init__(self, value, name="", local_grads={}):
    self.value = value
    self.local_grads = local_grads
    self.name = name
  def __repr__(self):
    return f'Variable "{self.name}"'

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
