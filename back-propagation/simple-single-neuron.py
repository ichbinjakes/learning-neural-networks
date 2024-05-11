# Analytic gradient calculation on simple graph. Based on:
# - http://karpathy.github.io/neuralnets/
# - https://www.deeplearningbook.org/contents/mlp.html

import math


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


class SigmoidNode:
    """
    Sigmoid graph operation
    z(x) = 1 / (1+e^-x)
    dz/dx = z(x)(1-z(x))
    """
    def __init__(self):
        self.x = None
        self.out = None

    @staticmethod
    def _sigmoid(x: int | float):
        return 1 / (1 + math.exp(-1 * x))

    def forward(self, x: Unit):
        self.x = x
        self.out = Unit(self._sigmoid(x.value), 0.0)
        return self.out

    def backward(self):
        val = self._sigmoid(self.x.value)
        self.x.gradient += val * (1 - val) * self.out.gradient


# example 1:
# g(ax + by + c), g is the sigmoid function.
# 
# x----|   |
#      | X |----|   |
# a----|___|    |   |    |   |
#               | + |----|   |
# y----|   |    |   |    |   |
#      | X |----|___|    | + |--0.5-| g |----z
# b----|___|             |   |
#                        |   |
# c----------------------|___|
# 


a = Unit(1, 0.0)
b = Unit(2, 0.0)
c = Unit(-3, 0.0)
x = Unit(-1, 0.0)
y = Unit(3, 0.0)

ax = MultiplicationNode()
by = MultiplicationNode()
ax_by = AdditionNode()
ax_by_c = AdditionNode()
output = SigmoidNode()


# ax_result = ax.forward(x, a)
# assert ax_result.value == -1

# by_result = by.forward(y, b)
# assert by_result.value == 6

# ax_by_result = ax_by.forward(ax_result, by_result)
# assert ax_by_result.value == 5

# ax_by_c_result = ax_by_c.forward(ax_by_result, c)
# assert ax_by_c_result.value == 2

# output_result = output.forward(ax_by_c_result)
# assert output_result.value > 0.88 and output_result.value < 0.89

def forward():
    ax_result = ax.forward(x, a)
    by_result = by.forward(y, b)
    ax_by_result = ax_by.forward(ax_result, by_result)
    ax_by_c_result = ax_by_c.forward(ax_by_result, c)
    output_result = output.forward(ax_by_c_result)
    return output_result

output_result = forward()

output_result.gradient = 1
for i in [output, ax_by_c, ax_by, by, ax]:
    i.backward()


step_size = 0.01
a.value += step_size * a.gradient
b.value += step_size * b.gradient
c.value += step_size * c.gradient
x.value += step_size * x.gradient
y.value += step_size * y.gradient

s = forward()
assert s.value > 0.8824 and s.value < 0.8826

