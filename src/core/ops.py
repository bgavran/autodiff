import re
from functools import reduce
from core.computational_graph import *


class Add(Operation):
    def __init__(self, *elems, name="Add"):
        super().__init__(list(elems), name)

    def eval(self, input_dict):
        arr = [elem(input_dict) for elem in self.children]
        # Using python sum instead of np.sum because python converts types correctly
        return sum(arr)

    @CompositeWrapper.from_graph_df
    def graph_df(self, wrt="", grad=None):
        wrt_count = [child.name for child in self.children].count(wrt)
        return grad * Constant(wrt_count)


class Negate(Operation):
    def __init__(self, node, name="Negate"):
        super().__init__([node], name)
        self.node = node

    def eval(self, input_dict):
        return -self.node(input_dict)

    @CompositeWrapper.from_graph_df
    def graph_df(self, wrt="", grad=None):
        if self.node.name == wrt:
            return -grad
        else:
            return 0


class Mul(Operation):
    def __init__(self, *elems, name="Mul"):
        if not elems:
            name = "1-Mul"
        super().__init__(list(elems), name)

    def eval(self, input_dict):
        arr = [elem(input_dict) for elem in self.children]
        # Using python's functions instead of numpy.prod since prod doesn't do type checking
        prod = 1
        for val in arr:
            prod *= val
        return prod

        # There are some operands that are just a number and here we broadcast them?
        # arr = [np.array([val]) if isinstance(val, numbers.Number) else val for val in arr]
        # return np.prod(arr, axis=0)

    @CompositeWrapper.from_graph_df
    def graph_df(self, wrt="", grad=None):
        wrt_count = [child.name for child in self.children].count(wrt)
        if wrt_count > 0:
            list_without_wrt = [child for child in self.children if child.name is not wrt]
            wrt_elem = next(child for child in self.children if child.name is wrt)
            return Mul(grad,
                       wrt_count,
                       Pow(wrt_elem, wrt_count - 1),
                       Mul(*list_without_wrt))
        return 0


class Recipr(Operation):
    def __init__(self, node, name="Reciprocal"):
        super().__init__([node], name)
        self.node = self.children[0]

    def eval(self, input_dict):
        return 1 / (self.node(input_dict) + Operation.epsilon)

    @CompositeWrapper.from_graph_df
    def graph_df(self, wrt="", grad=None):
        if self.node.name == wrt:
            return grad * -Recipr(self.node * self.node)
        return 0


class Transpose(Operation):
    def __init__(self, node, name="Transpose"):
        super().__init__([node], name)
        self.node = self.children[0]

    def eval(self, input_dict):
        return np.transpose(self.node(input_dict))

    def graph_df(self, wrt="", grad=None):
        if self.node.name == wrt:
            return Transpose(grad)
        return 0


class MatMul(CompositeOperation):
    def __init__(self, a, b, name="MatMul", expand_when_graphed=False):
        super().__init__([a, b], name, expand_when_graphed=expand_when_graphed)
        self.out = self.init_graph()

    def graph(self):
        return EinSum("ij,jk->ik", self.children[0], self.children[1])


class EinSum(Operation):
    def __init__(self, op_str, *operands, name="EinSum"):
        super().__init__(list(operands), name)
        self.op_str = op_str
        self.operands = self.children

        self.opnames = re.split(",|->", self.op_str)
        self.all_letters = "".join(set("".join(self.opnames[:-1])))

        assert len(self.operands) + 1 == len(self.opnames)

    def eval(self, input_dict):
        """
        Currently the problem is that some of the operands are just a number (like the input gradient)
        and they need to be broadcasted correctly to their shape.
        The shape can be inferred from all the other operands.

        But can it? For matmul the first dimension is never known.
        But perhaps that means that it shouldn't be possible to know it and that it should always
        be possible to broadcast the result?
        """
        arr = [op(input_dict) for op in self.operands]
        letter_to_dim = {}
        for i, val in enumerate(arr):
            if isinstance(val, np.ndarray):
                shape = val.shape
                letters = self.opnames[i]
                assert len(shape) == len(letters)
                for letter, dim in zip(letters, shape):
                    letter_to_dim[letter] = dim
                    # array_operands.append([val.shape, self.opnames[i]])

        for i, val in enumerate(arr):
            if isinstance(val, numbers.Number):
                shape_in_letters = self.opnames[i]
                shape_in_dims = [letter_to_dim.get(letter, 1) for letter in shape_in_letters]
                arr[i] = np.broadcast_to(val, shape_in_dims)

        return np.einsum(self.op_str, *arr)

    @CompositeWrapper.from_graph_df
    def graph_df(self, wrt="", grad=None):
        """
        Usual einsum operation looks something like this c = einsum("ij,kj->ik", a, b)
        df w.r.t. the first parameter just changes the op to look like this: df = einsum("ik,kj->ij", c, b).
        It basically just switches the output with one of the inputs.

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

            # this is concatenation of two lists in np array
            operands_with_grad = np.array([op for op in self.operands] + [grad])[order]

            opnames = self.opnames.copy()
            # opnames[-1] = self.all_letters
            opnames = EinSum.to_einsum_string(np.array(opnames)[order])

            return EinSum(opnames, *operands_with_grad[:-1])

        except NameError:
            print("wrt == ", wrt)
            return 0

    @staticmethod
    def to_einsum_string(list_of_ops):
        return ",".join(list_of_ops[:-1]) + "->" + list_of_ops[-1]


class Tanh(CompositeOperation):
    def __init__(self, node, name="Tanh", expand_when_graphed=False):
        super().__init__([node], name, expand_when_graphed=expand_when_graphed)
        self.out = self.init_graph()

    def graph(self):
        node = self.children[0]
        return 2 * Sigmoid(node) - 1


class Sigmoid(Operation):
    def __init__(self, node, name="Sigmoid"):
        super().__init__([node], name)
        self.node = self.children[0]

    def eval(self, input_dict):
        return 1.0 / (1.0 + np.exp(-self.node(input_dict)))

    @CompositeWrapper.from_graph_df
    def graph_df(self, wrt="", grad=None):
        if self.node.name == wrt:
            return grad * self * (1 - self)
        return 0


class ReLU(Operation):
    def __init__(self, node, name="ReLU"):
        super().__init__([node], name)
        self.node = self.children[0]

    def eval(self, input_dict):
        return self.bigger_than_zero(input_dict) * self.node(input_dict)

    @CompositeWrapper.from_graph_df
    def graph_df(self, wrt="", grad=None):
        return self * Recipr(self)

    def bigger_than_zero(self, input_dict):
        return self.node(input_dict) > 0


class Pow(Operation):
    def __init__(self, first, second, name="Pow"):
        super().__init__([first, second], name)
        self.first = self.children[0]
        self.second = self.children[1]

    def eval(self, input_dict):
        return np.power(self.first(input_dict), self.second(input_dict))

    @CompositeWrapper.from_graph_df
    def graph_df(self, wrt="", grad=None):
        if self.first.name == self.second.name == wrt:
            return self * (Log(self.first) + 1)
        elif self.first.name == wrt:
            return self.second * Pow(self.first, self.second - 1)
        elif self.second.name == wrt:
            return Log(self.first) * self
        return 0


class Log(Operation):
    def __init__(self, node, name="Log"):
        # Natural log
        super().__init__([node], name)
        self.node = self.children[0]

    def eval(self, input_dict):
        return np.log(self.node(input_dict))

    @CompositeWrapper.from_graph_df
    def graph_df(self, wrt="", grad=None):
        if self.node.name == wrt:
            return grad * Recipr(self.node)
        return 0


class CrossEntropy(CompositeOperation):
    def __init__(self, true, predicted, name="CrossEntropy"):
        super().__init__([true, predicted], name)

    def graph(self):
        true = self.children[0]
        predicted = self.children[1]

        return true * Log(Recipr(predicted)) + (1 - true) * Log(Recipr(1 - predicted))


class TestRecursivelyComposite(CompositeOperation):
    def __init__(self, node, count=1, name="Test_Composite", expand_when_graphed=True):
        super().__init__([node], name=name, expand_when_graphed=expand_when_graphed)
        self.count = count
        self.out = self.init_graph()

    def graph(self):
        node = self.children[0]
        t = Variable(name="t")

        if self.count:
            return node * (1 + TestRecursivelyComposite(node,
                                                        count=self.count - 1,
                                                        expand_when_graphed=self.expand_when_graphed))
        else:
            return SquaredDifference(node, t, expand_when_graphed=True)


class Exp(Operation):
    def __init__(self, node, name="Exp"):
        super().__init__([node], name)
        self.node = self.children[0]

    def eval(self, input_dict):
        return np.exp(self.node(input_dict))

    @CompositeWrapper.from_graph_df
    def graph_df(self, wrt, grad=None):
        if self.node.name == wrt:
            return grad * self
        return 0


class SquaredDifference(CompositeOperation):
    def __init__(self, first, second, name="Squared_diff", expand_when_graphed=True):
        super().__init__([first, second], name=name, expand_when_graphed=expand_when_graphed)
        self.out = self.init_graph()

    def graph(self):
        first = self.children[0]
        second = self.children[1]

        diff = first - second
        return diff * diff


class Grad(CompositeOperation):
    def __init__(self, node, wrt, initial_grad=None, name="Grad", expand_when_graphed=True):
        super().__init__([node],
                         name=name + " w.r.t. " + str(wrt),
                         expand_when_graphed=expand_when_graphed)
        self.wrt = wrt
        # c = np.array([1])
        c = 1
        self.initial_grad = Constant(c, name=node.name + "_grad") if initial_grad is None else initial_grad
        self.out = self.init_graph()

    def graph(self):
        out_node = self.children[0].get_node()
        nodes = out_node.topo_sort()

        try:
            wrt_node = next(node for node in nodes if node.name == self.wrt)
        except StopIteration:
            raise ValueError("Node with the name \"" + str(self.wrt) + "\" is not in the graph!")

        dct = collections.defaultdict(list)
        dct[out_node.id].append(self.initial_grad)

        for node in reversed(nodes):
            dct[node.id] = Add(*dct[node.id], name=node.name + "_grad_sum")

            # Don't evaluate the same child twice!
            # The node might be composite, in which case we need the children from its .out variable
            for child in set(node.get_node().children):
                dct[child.id].append(node.graph_df(child.name, grad=dct[node.id]))

        return dct[wrt_node.id]
