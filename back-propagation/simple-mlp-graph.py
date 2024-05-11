# Analytic gradient calculation on simple graph. Based on:
# - http://karpathy.github.io/neuralnets/
# - https://www.deeplearningbook.org/contents/mlp.html


class Unit:
    def __init__(self, value, gradient) -> None:
        self.value = value
        self.gradient = gradient


class AdditionNode:
    """
    Addition graph operation.
    z = a + b
    dz/da = 1
    dz/db = 1
    """
    def __init__(self):
        self.a = None
        self.b = None
        self.out = None

    def forward(self, a: Unit, b: Unit):
        self.a = a
        self.b = b
        self.out = Unit(a.value + b.value, 0.0)
        return self.out
    
    def backward(self):
        self.a.gradient += 1 * self.out.gradient
        self.b.gradient += 1 * self.out.gradient


class MultiplicationNode:
    """
    Multiplication graph operation.
    z = x * y
    dz/dx = y
    dz/dy = x
    """

    def __init__(self):
        self.a = None
        self.b = None
        self.out = None

    def forward(self, a: Unit, b: Unit):
        self.a = a
        self.b = b
        self.out = Unit(a.value * b.value, 0.0)
        return self.out
    
    def backward(self):
        self.a.gradient += self.b.value * self.out.gradient
        self.b.gradient += self.a.value * self.out.gradient


# example 1:
# 
# x1  --|(a)|
#       | X | ---|(e)|
# w11 --|___|    |   | ------ |(g)|     |(i)|
#                | + |        | X | ----|   |
# x2  --|(b)|    |   |  h1 -- |___|     |   |
#       | X | ---|___|                  |   |
# w21 --|___|                           |   |
#                                       | + | -- z -- (add activation)
# x1  --|(c)|                           |   |
#       | X | ---|(f)|                  |   |
# w12 --|___|    |   | ------ |(h)|     |   |
#                | + |        | X | ----|   |
# x2  --|(d)|    |   |  h2 -- |___|     |___|
#       | X | ---|___|
# w22 --|___|   


x1 = Unit(-2, 0.0)
x2 = Unit(4, 0.0)

w11 = Unit(1, 0.0)
w21 = Unit(2, 0.0)
w12 = Unit(3, 0.0)
w22 = Unit(2, 0.0)

h1 = Unit(1, 0.0)
h2 = Unit(3, 0.0)

a = MultiplicationNode()
b = MultiplicationNode()
c = MultiplicationNode()
d = MultiplicationNode()

e = AdditionNode()
f = AdditionNode()

g = MultiplicationNode()
h = MultiplicationNode()

i = AdditionNode()

# Forward pass
x1w11 = a.forward(x1, w11)
assert x1w11.value == -2
x2w21 = b.forward(x2, w21)
assert x2w21.value == 8
x1w12 = c.forward(x1, w12)
assert x1w12.value == -6
x2w22 = d.forward(x2, w22)
assert x2w22.value == 8

e_result = e.forward(x1w11, x2w21)
assert e_result.value == 6
f_result = f.forward(x1w12, x2w22)
assert f_result.value == 2

g_result = g.forward(e_result, h1)
assert g_result.value == 6
h_result = h.forward(f_result, h2)
assert h_result.value == 6

z = i.forward(g_result, h_result)
assert z.value == 12

# backwards pass
z.gradient = 1

for i in [i, h, g, f, e, d, c, b, a]:
    i.backward()

assert h2.gradient == 2
assert h1.gradient == 6

assert w11.gradient == x1.value
assert w21.gradient == x2.value
assert w12.gradient == 3 * x1.value
assert w22.gradient == 3 * x2.value