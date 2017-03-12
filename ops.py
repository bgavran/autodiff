from ComputationalGraph import *


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


class Subtract(Operation):
    def __init__(self, children, name=""):
        assert len(children) == 2
        super().__init__(children, name)

    def f(self, input_dict):
        return self.children[0](input_dict) - self.children[1](input_dict)

    def df(self, input_dict, wrt=""):
        """
        Returns the number of arguments that have the same name as wrt, 0 otherwise
        :param wrt:
        :param input_dict:
        :return:
        """
        if wrt == self.children[0].name:
            return 1
        elif wrt == self.children[1].name:
            return -1
        return 0


class Add(Operation):
    def __init__(self, children, name=""):
        assert len(children) == 2
        super().__init__(children, name)

    def f(self, input_dict):
        return self.children[0](input_dict) + self.children[1](input_dict)

    def df(self, input_dict, wrt=""):
        """
        Returns the number of arguments that have the same name as wrt, 0 otherwise
        :param wrt:
        :param input_dict:
        :return:
        """
        return [child.name for child in self.children].count(wrt)


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
