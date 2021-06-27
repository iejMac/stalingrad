import numpy as np

from tensor import Tensor

x = Tensor(np.random.uniform(size=(3, 3)) - 0.5)
print(x)

y = x.relu()
print(y)

y.backward()
print(x.grad)


# grad = y.func.backward(y.func, np.ones((3, 3)))
# print(grad)

'''    
x1 = Tensor()
x2 = Tensor()
x3 = Tensor()
y = Tensor()

x12 = Add(x1, x2)
x23 = Sub(x2, x3)

x123 = Mul(x12, x23)
E = Sub(x123, y)
Loss = Pow(E, 2)

Loss.backward()
'''
