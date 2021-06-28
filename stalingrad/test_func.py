import numpy as np

from tensor import Tensor

x1 = Tensor(np.arange(3).reshape((3, 1)).astype(float))
x2 = Tensor(np.ones((1, 3))*-1)

y_ = x1 @ x2

y_.backward()

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
