import variable
from variable import Variable

a = Variable(2, name="a")
b = Variable(3, name="b")

# c = a*b
c = a*b
# d = a+b
d = a-b

e = c*d

label = Variable(5, name="lab")

mse_loss = (e-label)**2

print(mse_loss)
print(mse_loss.value)
