from core.computational_graph import *


class Add(Operation):
    def __init__(self, first, second, name="Add"):
        super().__init__([first, second], name)
        self.first = self.children[0]
        self.second = self.children[1]

    def f(self, input_dict):
        return self.first(input_dict) + self.second(input_dict)

    def df(self, input_dict, wrt="", grad=None):
        """
        Returns the array of ones the size of the wrt argument (if it exists in children)
        The previous count method:
        return [child.name for child in self.children].count(wrt)
        had a problem where, in a really simple graph with out = w0 + w1 the shape of meshgrid wouldn't be propagated
        :param input_dict:
        :param wrt:
        :param grad:
        :return:
        """
        if self.first.name == wrt:
            return grad*np.ones_like(self.first(input_dict))
        elif self.second.name == wrt:
            return grad*np.ones_like(self.second(input_dict))
        else:
            return 0


class Negate(Operation):
    def __init__(self, node, name="Negate"):
        super().__init__([node], name)
        self.node = node

    def f(self, input_dict):
        return -self.node(input_dict)

    def df(self, input_dict, wrt="", grad=None):
        if self.node.name == wrt:
            return -grad*np.ones_like(self.node(input_dict))
        else:
            return 0


class Mul(Operation):
    def __init__(self, first, second, name="Mul"):
        super().__init__([first, second], name)
        self.first = self.children[0]
        self.second = self.children[1]

    def f(self, input_dict):
        return self.first(input_dict) * self.second(input_dict)

    def df(self, input_dict, wrt="", grad=None):
        if wrt == self.first.name:
            return grad * self.second(input_dict)
        elif wrt == self.second.name:
            return grad * self.first(input_dict)
        return 0


class Recipr(Operation):
    def __init__(self, node, name="Reciprocal"):
        super().__init__([node], name)
        self.node = self.children[0]

    def f(self, input_dict):
        return 1 / self.node(input_dict)

    def df(self, input_dict, wrt="", grad=None):
        if self.node.name == wrt:
            val = self.node(input_dict)
            return grad * -1 / (val * val)
        return 0


class Transpose(Operation):
    def __init__(self, node, name="Transpose"):
        super().__init__([node], name)
        self.node = self.children[0]

    def f(self, input_dict):
        return np.transpose(self.node(input_dict))

    def df(self, input_dict, wrt="", grad=None):
        return np.transpose(grad)


class MatMul(CompositeOperation):
    def __init__(self, a, b, name="MatMul"):
        super().__init__([a, b], name)
        self.a = self.children[0]
        self.b = self.children[1]
        self.graph()

    def graph(self):
        self.out = EinSum("ij,jk->ik", self.a, self.b)


class EinSum(Operation):
    def __init__(self, op_str, *operands, name="EinSum"):
        super().__init__(operands, name)
        self.op_str = op_str
        self.operands = self.children

        import re
        self.opnames = re.split(",|->", self.op_str)
        self.all_letters = "".join(set("".join(self.opnames[:-1])))

        assert len(self.operands) + 1 == len(self.opnames)

    def f(self, input_dict):
        return np.einsum(self.op_str, *[op(input_dict) for op in self.operands])

    def df(self, input_dict, wrt="", grad=None):
        """

        :param input_dict:
        :param wrt:
        :param grad:
        :return:
        """
        for i, operand in enumerate(self.operands):
            if operand.name == wrt:
                loc = i
                break
        try:
            order = list(range(len(self.opnames)))
            order[loc], order[-1] = order[-1], order[loc]

            operands_with_grad = np.array([op(input_dict) for op in self.operands] + [grad])[order]

            opnames = self.opnames.copy()
            # opnames[-1] = self.all_letters
            opnames = EinSum.to_einsum_string(np.array(opnames)[order])

            return np.einsum(opnames, *operands_with_grad[:-1])

        except NameError:
            print("wrt == ", wrt)
            return 0

    @staticmethod
    def to_einsum_string(list_of_ops):
        return ",".join(list_of_ops[:-1]) + "->" + list_of_ops[-1]


class Sigmoid(Operation):
    def __init__(self, node, name="Sigmoid"):
        super().__init__([node], name)
        self.node = self.children[0]

    def f(self, input_dict):
        return 1 / (1 + np.exp(-self.node(input_dict)))

    def df(self, input_dict, wrt="", grad=None):
        return grad * self.f(input_dict) * (1 - self.f(input_dict))


class ReLU(Operation):
    def __init__(self, node, name="ReLU"):
        super().__init__([node], name)
        self.node = self.children[0]

    def f(self, input_dict):
        return self.bigger_than_zero(input_dict) * self.node(input_dict)

    def df(self, input_dict, wrt="", grad=None):
        if wrt == self.node.name:
            return grad*self.bigger_than_zero(input_dict)

    def bigger_than_zero(self, input_dict):
        return self.node(input_dict) > 0


class Gauss(Operation):
    def __init__(self, node, name="Gauss"):
        super().__init__([node], name)
        self.node = self.children[0]

    def f(self, input_dict):
        z = 1 / np.sqrt(2*np.pi)
        return z * np.exp(-0.5 * np.square(self.node(input_dict)))

    def df(self, input_dict, wrt="", grad=None):
        return grad * -self.node(input_dict) * self(input_dict)


class SquaredDifference(CompositeOperation):
    def __init__(self, first, second, name="Squared difference"):
        super().__init__([first, second], name)
        self.first = self.children[0]
        self.second = self.children[1]
        self.graph()

    def graph(self):
        diff = self.first - self.second
        self.out = diff * diff
