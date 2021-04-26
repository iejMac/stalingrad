import variable
from variable import Variable
from variable import mul, add, neg, inv

a = Variable(2, name="a")
b = Variable(3, name="b")

# c = a*b
c = a*b
# d = a+b
d = a-b

# e = c*d = a*b*(a+b) = a^2b + b^2a = 12 + 18 = 30
e = c*d

f = e/d

print(f.value)
