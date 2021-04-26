import variable
from variable import Variable
from variable import mul, add

a = Variable(2, name="a")
b = Variable(3, name="b")

# c = a*b
c = mul(a, b)
# d = a+b
d = add(a, b)

# e = c*d = a*b*(a+b) = a^2b + b^2a = 12 + 18 = 30
e = mul(c, d)

print(e)
print(e.value)
print(e.local_grads[c])
