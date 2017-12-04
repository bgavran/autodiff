import time
import numbers
import numpy as np
from contextlib import contextmanager


class Node:
    epsilon = 1e-12
    id = 0
    context_list = []

    def __init__(self, children, name="Node"):
        # wraps normal numbers into Variables
        self.children = [child if isinstance(child, Node) else Variable(child) for child in children]
        self.name = name
        self.cached = None
        self.shape = None

        self.context_list = Node.context_list.copy()
        self.id = Node.id
        Node.id += 1

    def _eval(self):
        """

        :return: returns the value of the evaluated Node
        """
        raise NotImplementedError()

    def _partial_derivative(self, wrt, previous_grad):
        """
        Method which calculates the partial derivative of self with respect to the wrt Node.
        By defining this method without evaluation of any nodes, higher-order gradients
        are available for free.

        :param wrt: instance of Node, partial derivativative with respect to it
        :param previous_grad: gradient with respect to self
        :return: an instance of Node whose evaluation yields the partial derivative
        """
        raise NotImplementedError()

    def eval(self):
        if self.cached is None:
            self.cached = self._eval()

        return self.cached

    def partial_derivative(self, wrt, previous_grad):
        with add_context(self.name + " PD" + " wrt " + str(wrt)):
            return self._partial_derivative(wrt, previous_grad)

    def plot_comp_graph(self, view=True, name="comp_graph"):
        from ..visualization import graph_visualization
        graph_visualization.plot_comp_graph(self, view=view, name=name)

    def __call__(self, *args, **kwargs):
        return self.eval()

    def __str__(self):
        return self.name  # + " " + str(self.id)

    def __add__(self, other):
        from .ops import Add
        return Add(self, other)

    def __neg__(self):
        from .ops import Negate
        return Negate(self)

    def __sub__(self, other):
        return self.__add__(other.__neg__())

    def __rsub__(self, other):
        return self.__neg__().__add__(other)

    def __mul__(self, other):
        from .ops import Mul
        return Mul(self, other)

    def __matmul__(self, other):
        from .high_level_ops import MatMul
        return MatMul(self, other)

    def __rmatmul__(self, other):
        from .high_level_ops import MatMul
        return MatMul(other, self)

    def __imatmul__(self, other):
        return self.__matmul__(other)

    def __truediv__(self, other):
        from .ops import Recipr
        return self.__mul__(Recipr(other))

    def __rtruediv__(self, other):
        from .ops import Recipr
        return Recipr(self).__mul__(other)

    def __pow__(self, power, modulo=None):
        from .ops import Pow
        return Pow(self, power)

    __rmul__ = __mul__
    __radd__ = __add__

    def __getitem__(self, item):
        from .reshape import Slice
        return Slice(self, item)


class Variable(Node):
    def __init__(self, value, name=None):
        if name is None:
            name = str(value)  # this op is really slow for np.arrays?!
        super().__init__([], name)

        if isinstance(value, numbers.Number):
            self._value = np.array(value, dtype=np.float64)
        else:
            self._value = value
        self.shape = self._value.shape

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        self.cached = self._value = val

    def _eval(self):
        return self._value

    def _partial_derivative(self, wrt, previous_grad):
        if self == wrt:
            return previous_grad
        return 0


@contextmanager
def add_context(ctx):
    Node.context_list.append(ctx + "_" + str(time.time()))
    try:
        yield
    finally:
        del Node.context_list[-1]
