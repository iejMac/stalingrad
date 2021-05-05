import variable
from variable import Variable

a = Variable(2, name="a")
b = Variable(3, name="b")
y = Variable(5, name="y")

# 6 = 2*3
c = a*b
# -1 = 2 - 3
d = a-b

# -6 = 6 * (-1)
e = c*d

# 121 = (-6 - 5)**2
mse_loss = (e - y)**2

assert(mse_loss.value == 121.0)
print("Loss Value Correct")

mse_loss.backprop(True)
print("MSE: ")
print(mse_loss.grad)
print(mse_loss.local_grads)

print("e: ")
print(e.grad)
print(e.local_grads)

print("c: ")
print(c.grad)
print(c.local_grads)

print("d: ")
print(d.grad)
print(d.local_grads)

print("a: ")
print(a.grad)
print(a.local_grads)

print("b: ")
print(b.grad)
print(b.local_grads)
