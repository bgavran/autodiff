import re
from core.computational_graph import *


class Identity(Primitive):
    def __init__(self, node, name="Identity"):
        super().__init__([node], name)
        self.node = self.children[0]

    def eval(self):
        return self.node.eval()

    def graph_df(self, wrt, grad):
        if self.node == wrt:
            return grad * self.node.graph_df(wrt, grad)
        return 0


class Mul(Primitive):
    def __init__(self, *elems, name="Mul"):
        if not elems:
            name = "1-" + name
        super().__init__(list(elems), name)

    def eval(self):
        # Using python's functions instead of numpy.prod since prod doesn't do type checking
        prod = 1
        for elem in self.children:
            prod = np.multiply(prod, elem())
        return prod

    @CompositeWrapper.from_graph_df
    def graph_df(self, wrt, grad):
        add_list = []
        for loc, child in enumerate(self.children):
            if child == wrt:
                add_list.append(Mul(*[ch for i, ch in enumerate(self.children) if loc != i]))
        return grad * Add(*add_list)


class Negate(Primitive):
    def __init__(self, node, name="Negate"):
        super().__init__([node], name)
        self.node = node

    def eval(self):
        return -self.node()

    @CompositeWrapper.from_graph_df
    def graph_df(self, wrt, grad):
        if self.node == wrt:
            return -grad
        else:
            return 0


class Recipr(Primitive):
    def __init__(self, node, name="Reciprocal"):
        """
        Elementwise reciprocal

        """
        super().__init__([node], name)
        self.node = self.children[0]

    def eval(self):
        val = self.node()
        return 1 / (val + Primitive.epsilon)

    @CompositeWrapper.from_graph_df
    def graph_df(self, wrt, grad):
        if self.node == wrt:
            return - grad * self * self
        return 0


class Transpose(Primitive):
    def __init__(self, node, name="Transpose"):
        super().__init__([node], name)
        self.node = self.children[0]

    def eval(self):
        return np.transpose(self.node())

    def graph_df(self, wrt, grad):
        if self.node == wrt:
            return Transpose(grad)
        return 0


class MatMul(CompositeOperation):
    def __init__(self, a, b, name="MatMul", graph_expand=True):
        super().__init__([a, b], name, graph_expand=graph_expand)
        self.out = self.init_graph()

    def graph(self):
        return EinSum("ij,jk->ik", self.children[0], self.children[1])


class EinSum(Primitive):
    def __init__(self, op_str, *operands, name="EinSum"):
        super().__init__(list(operands), name)
        self.op_str = op_str
        self.operands = self.children

        self.opnames = re.split(",|->", self.op_str)
        self.all_letters = "".join(set("".join(self.opnames[:-1])))

        assert len(self.operands) + 1 == len(self.opnames)

    def eval(self):
        """
        Currently the problem is that some of the operands are just a number (like the input gradient)
        and they need to be broadcasted correctly to their shape.
        The shape can be inferred from all the other operands.

        But can it? For matmul the first dimension is never known.
        But perhaps that means that it shouldn't be possible to know it and that it should always
        be possible to broadcast the result?
        """
        arr = [op() for op in self.operands]
        letter_to_dim = {}
        for i, val in enumerate(arr):
            if isinstance(val, np.ndarray):
                shape = val.shape
                letters = self.opnames[i]
                assert len(shape) == len(letters)
                for letter, dim in zip(letters, shape):
                    letter_to_dim[letter] = dim

        for i, val in enumerate(arr):
            if isinstance(val, numbers.Number):
                shape_in_letters = self.opnames[i]
                shape_in_dims = [letter_to_dim.get(letter, 1) for letter in shape_in_letters]
                arr[i] = np.broadcast_to(val, shape_in_dims)

        return np.einsum(self.op_str, *arr)

    @CompositeWrapper.from_graph_df
    def graph_df(self, wrt, grad):
        """
        Usual einsum operation looks something like this c = einsum("ij,kj->ik", a, b)
        df w_val.r.t. the first parameter just changes the op to look like this: df = einsum("ik,kj->ij", c, b).
        It basically just switches the output with one of the inputs.

        :param wrt:
        :param grad:
        :return:
        """
        for i, operand in enumerate(self.operands):
            if operand == wrt:
                loc = i
                break
        try:
            order = list(range(len(self.opnames)))
            order[loc], order[-1] = order[-1], order[loc]

            # this is concatenation of two lists in np array
            operands_with_grad = np.array([op for op in self.operands] + [grad])[order]

            opnames = self.opnames.copy()
            # opnames[-1] = self.all_letters
            opnames = EinSum.to_einsum_string(np.array(opnames)[order])

            return EinSum(opnames, *operands_with_grad[:-1])

        except NameError:
            print("wrt == ", wrt.name)
            return 0

    @staticmethod
    def to_einsum_string(list_of_ops):
        return ",".join(list_of_ops[:-1]) + "->" + list_of_ops[-1]


class ReLU(Primitive):
    def __init__(self, node, name="ReLU"):
        super().__init__([node], name)
        self.node = self.children[0]

    def eval(self):
        return self.bigger_than_zero() * self.node()

    @CompositeWrapper.from_graph_df
    def graph_df(self, wrt, grad):
        # TODO higher order gradient doesn't seem to be correct?
        # probably fixed now, but too slow to check?
        if self.node == wrt:
            return grad * self * Recipr(self)
        return 0

    def bigger_than_zero(self):
        return self.node() > 0


class Pow(Primitive):
    def __init__(self, first, second, name="Pow"):
        super().__init__([first, second], name)
        self.first = self.children[0]
        self.second = self.children[1]

    def eval(self):
        f = self.first()
        s = self.second()

        return np.power(f, s)

    @CompositeWrapper.from_graph_df
    def graph_df(self, wrt, grad):
        if self.first == self.second == wrt:
            return grad * self * (Log(self.first) + 1)
        elif self.first == wrt:
            return grad * self.second * Pow(self.first, self.second - 1)
        elif self.second == wrt:
            return grad * Log(self.first) * self
        return 0


class Log(Primitive):
    def __init__(self, node, name="Log"):
        super().__init__([node], name)
        self.node = self.children[0]

    def eval(self):
        return np.log(self.node() + Primitive.epsilon)

    @CompositeWrapper.from_graph_df
    def graph_df(self, wrt, grad):
        if self.node == wrt:
            return grad * Recipr(self.node)
        return 0


class Exp(Primitive):
    def __init__(self, node, name="Exp"):
        super().__init__([node], name)
        self.node = self.children[0]

    def eval(self):
        return np.exp(self.node())

    @CompositeWrapper.from_graph_df
    def graph_df(self, wrt, grad):
        if self.node == wrt:
            return grad * self
        return 0


class Tanh(CompositeOperation):
    def __init__(self, node, name="Tanh", graph_expand=False):
        super().__init__([node], name, graph_expand=graph_expand)
        self.out = self.init_graph()

    def graph(self):
        node = self.children[0]
        return 2 * Sigmoid(2 * node) - 1


class Sigmoid(CompositeOperation):
    def __init__(self, node, name="Sigmoid", graph_expand=False):
        super().__init__([node], name=name, graph_expand=graph_expand)
        self.node = self.children[0]
        self.out = self.init_graph()

    def graph(self):
        return 1 / (1 + Exp(-self.node))

    # This is not needed, but is a simplification?
    @CompositeWrapper.from_graph_df
    def graph_df(self, wrt, grad):
        if wrt == self.node:
            return grad * self * (1 - self)
        return 0


class SquaredDifference(CompositeOperation):
    def __init__(self, first, second, name="Squared_diff", graph_expand=True):
        super().__init__([first, second], name=name, graph_expand=graph_expand)
        self.out = self.init_graph()

    def graph(self):
        first = self.children[0]
        second = self.children[1]

        diff = first - second
        return diff * diff


class TestRecursivelyComposite(CompositeOperation):
    def __init__(self, node, count=1, name="Test_Composite", graph_expand=True):
        super().__init__([node], name=name, graph_expand=graph_expand)
        self.count = count
        self.out = self.init_graph()

    def graph(self):
        node = self.children[0]
        t = Variable(7, name="t")

        if self.count:
            return node * (1 + TestRecursivelyComposite(node,
                                                        count=self.count - 1,
                                                        graph_expand=self.graph_expand))
        else:
            return SquaredDifference(node, t, graph_expand=True)
