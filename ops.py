from ComputationalGraph import *


class Add(Operation):
    def __init__(self, first, second, name=""):
        super().__init__([first, second], name)
        self.first = self.children[0]
        self.second = self.children[1]

    def f(self, input_dict):
        return self.first(input_dict) + self.second(input_dict)

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
        if self.first.name == wrt:
            return np.ones_like(self.first(input_dict))
        elif self.second.name == wrt:
            return np.ones_like(self.second(input_dict))
        else:
            return 0


class Negate(Operation):
    def __init__(self, node, name=""):
        super().__init__([node], name)
        self.node = node

    def f(self, input_dict):
        return -self.node(input_dict)

    def df(self, input_dict, wrt=""):
        if self.node.name == wrt:
            return -np.ones_like(self.node(input_dict))
        else:
            return 0


class Mul(Operation):
    def __init__(self, first, second, name=""):
        super().__init__([first, second], name)
        self.first = self.children[0]
        self.second = self.children[1]

    def f(self, input_dict):
        return self.first(input_dict) * self.second(input_dict)

    def df(self, input_dict, wrt=""):
        if wrt == self.first.name:
            return self.second(input_dict)
        elif wrt == self.second.name:
            return self.first(input_dict)
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


class Recipr(Operation):
    def __init__(self, node, name=""):
        super().__init__([node], name)
        self.node = self.children[0]

    def f(self, input_dict):
        return 1 / self.node(input_dict)

    def df(self, input_dict, wrt=""):
        if self.node.name == wrt:
            val = self.node(input_dict)
            return -1 / (val * val)
        return 0


class Sigmoid(Operation):
    def __init__(self, node, name=""):
        super().__init__([node], name)
        self.node = self.children[0]

    def f(self, input_dict):
        return 1 / (1 + np.exp(-self.node(input_dict)))

    def df(self, input_dict, wrt=""):
        return self.f(input_dict) * (1 - self.f(input_dict))


class SquareDiff(Operation):
    def __init__(self, first, second, name=""):
        super().__init__([first, second], name)
        self.first = self.children[0]
        self.second = self.children[1]

    def f(self, input_dict):
        return np.square(self.first(input_dict) - self.second(input_dict)) / 2

    def df(self, input_dict, wrt=""):
        return self.first(input_dict) - self.second(input_dict)


class Gauss(Operation):
    def __init__(self, node, name=""):
        super().__init__([node], name)
        self.node = self.children[0]

    def f(self, input_dict):
        return np.exp(-np.square(self.node(input_dict)))

    def df(self, input_dict, wrt=""):
        return -2 * self.node(input_dict) * np.exp(-np.square(self.node(input_dict)))
