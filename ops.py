from ComputationalGraph import *


class Add(Operation):
    def __init__(self, children, name=""):
        assert len(children) == 2
        super().__init__(children, name)

    def f(self, input_dict):
        return self.children[0](input_dict) + self.children[1](input_dict)

    def df(self, input_dict, wrt=""):
        """
        Returns the array of ones the size of the wrt argument (if it exists in children)
        The previous count method:
        return [child.name for child in self.children].count(wrt)
        had a problem where, in a really simple graph with out = w0 + w1 the shape of meshgrid wouldn't be propagated
        :param wrt:
        :param input_dict:
        :return:
        """
        if self.children[0].name == wrt:
            return np.ones_like(self.children[0](input_dict))
        elif self.children[1].name == wrt:
            return np.ones_like(self.children[1](input_dict))
        else:
            return 0


class Negate(Operation):
    def __init__(self, children, name=""):
        assert len(children) == 1
        super().__init__(children, name)

    def f(self, input_dict):
        return -self.children[0](input_dict)

    def df(self, input_dict, wrt=""):
        if self.children[0].name == wrt:
            return -np.ones_like(self.children[0](input_dict))
        else:
            return 0


class Mul(Operation):
    def __init__(self, children, name=""):
        assert len(children) == 2
        super().__init__(children, name)

    def f(self, input_dict):
        return self.children[0](input_dict) * self.children[1](input_dict)

    def df(self, input_dict, wrt=""):
        if wrt == self.children[0].name:
            return self.children[1](input_dict)
        elif wrt == self.children[1].name:
            return self.children[0](input_dict)
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


class Sigmoid(Operation):
    def __init__(self, children, name=""):
        assert len(children) == 1
        super().__init__(children, name)

    def f(self, input_dict):
        return 1 / (1 + np.exp(-self.children[0](input_dict)))

    def df(self, input_dict, wrt=""):
        return self.f(input_dict) * (1 - self.f(input_dict))


class SquareCost(Operation):
    def __init__(self, children, name=""):
        assert len(children) == 2
        super().__init__(children, name)

    def f(self, input_dict):
        return np.square(self.children[0](input_dict) - self.children[1](input_dict)) / 2

    def df(self, input_dict, wrt=""):
        return self.children[0](input_dict) - self.children[1](input_dict)


class Gauss(Operation):
    def __init__(self, children, name=""):
        assert len(children) == 1
        super().__init__(children, name)

    def f(self, input_dict):
        return np.exp(-np.square(self.children[0](input_dict)))

    def df(self, input_dict, wrt=""):
        return -2 * self.children[0](input_dict) * np.exp(-np.square(self.children[0](input_dict)))
