import unittest
from stalingrad.variable import Variable

class TestAutodiff(unittest.TestCase):
  def test_autodiff(self):
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

    self.assertEqual(mse_loss.value, 121.0)

    mse_loss.backprop()

    self.assertEqual(a.grad, -66.0)
    self.assertEqual(b.grad, 176.0)

if __name__ == "__main__":
  unittest.main()
