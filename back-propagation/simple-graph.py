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
# x --|   |
#     | X | -- z
# y --|___|
# 

graph = MultiplicationNode()
x = Unit(2, 0.0)
y = Unit(-3, 0.0)

z = graph.forward(x, y)

assert z.value == -6

# graph.out.gradient = 1
z.gradient = 1
graph.backward()
assert x.gradient == -3
assert y.gradient == 2

# example 2:
# 
# x --|   |    |   |
#     | X | -- |   |
# y --|___|    | + | -- z
#              |   |
#         c -- |___|

graph_a = MultiplicationNode()
graph_b = AdditionNode()

x = Unit(4, 0.0)
y = Unit(-2, 0.0)
c = Unit(3, 0.0)

xy = graph_a.forward(x, y)
z = graph_b.forward(xy, c)
assert z.value == -5

z.gradient = 1
graph_b.backward()
assert c.gradient == 1

graph_a.backward()
assert x.gradient == -2
assert y.gradient == 4
