import numpy as np
from ComputationalGraph import *


class Mul(Operation):
    def __init__(self, children, name=""):
        assert len(children) == 2
        super().__init__(children, name)

    def f(self):
        return self.children[0]() * self.children[1]()

    def df(self, wrt=""):
        if wrt == self.children[0].name:
            return self.children[1]()
        elif wrt == self.children[1].name:
            return self.children[0]()
        return 0
        # """
        # Can be differentiated even if both inputs are the wrt argument. Should work also when neither is.
        # :param wrt:
        # :return:
        # """
        # res = 0
        # if wrt == self.children[0].name:
        #     res += self.children[1]()
        # if wrt == self.children[1].name:
        #     res += self.children[0]()
        # return res


class Add(Operation):
    def __init__(self, children, name=""):
        assert len(children) == 2
        super().__init__(children, name)

    def f(self):
        return self.children[0]() + self.children[1]()

    def df(self, wrt=""):
        """
        Returns the number of arguments that have the same name as wrt, 0 otherwise
        :param wrt:
        :return:
        """
        return [child.name for child in self.children].count(wrt)


class Sigmoid(Operation):
    def __init__(self, children, name=""):
        assert len(children) == 1
        super().__init__(children, name)

    def f(self):
        return 1 / (1 + np.exp(-self.children[0]()))

    def df(self, wrt=""):
        return self.f() * (1 - self.f())


class SquareCost(Operation):
    def __init__(self, children, name=""):
        assert len(children) == 2
        super().__init__(children, name)

    def f(self):
        return np.square(self.children[0]() - self.children[1]()) / 2

    def df(self, wrt=""):
        return self.children[0]() - self.children[1]()


class Gauss(Operation):
    def __init__(self, children, name=""):
        assert len(children) == 1
        super().__init__(children, name)

    def f(self):
        return np.exp(-np.square(self.children[0]()))

    def df(self, wrt=""):
        return -2 * self.children[0]() * np.exp(-np.square(self.children[0]()))
