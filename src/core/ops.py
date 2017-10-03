import re
from core.computational_graph import *


class Identity(Primitive):
    def __init__(self, node, name="Identity"):
        super().__init__([node], name)
        self.node = self.children[0]

    def f(self):
        return self.node.f()

    def graph_df(self, wrt, grad):
        if self.node == wrt:
            return grad * self.node.graph_df(wrt, grad)
        return 0


class Mul(Primitive):
    def __init__(self, *elems, name="Mul"):
        if not elems:
            name = "1-" + name
        super().__init__(list(elems), name)

    def f(self):
        # Using python's functions instead of numpy.prod since prod doesn't do type checking
        prod = 1
        for elem in self.children:
            prod = np.multiply(prod, elem())
        return prod

    @module_wrapper
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

    def f(self):
        return -self.node()

    @module_wrapper
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

    def f(self):
        val = self.node()
        return 1 / (val + Primitive.epsilon)

    @module_wrapper
    def graph_df(self, wrt, grad):
        if self.node == wrt:
            return - grad * self * self
        return 0


class Transpose(Primitive):
    def __init__(self, node, name="Transpose"):
        super().__init__([node], name)
        self.node = self.children[0]

    def f(self):
        return np.transpose(self.node())

    def graph_df(self, wrt, grad):
        if self.node == wrt:
            return Transpose(grad)
        return 0


class MatMul(Module):
    def __init__(self, a, b, name="MatMul", expand_graph=True):
        super().__init__([a, b], name, expand_graph=expand_graph)
        self.out = self.init_graph()

    def graph(self):
        return Einsum("ij,jk->ik", self.children[0], self.children[1])


class Einsum(Primitive):
    def __init__(self, op_str, *operands, name="EinSum"):
        super().__init__(list(operands), name + " " + op_str)
        # TODO what if in the inputs there's two same variables?
        # TODO what if there's ellipsis in the op_str?
        self.op_str = op_str
        self.operands = self.children

        self.opnames = re.split(",|->", self.op_str)
        self.all_letters = "".join(set("".join(self.opnames[:-1])))

        assert len(self.operands) + 1 == len(self.opnames)

    def f(self):
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

    @module_wrapper
    def graph_df(self, wrt, grad):
        """
        Usual einsum operation looks something like this c = einsum("ij,kj->ik", a, b)
        Gradient w.r.t. the first parameter just changes the op to look like this: df = einsum("ik,kj->ij", c, b).
        It basically just switches the output with one of the inputs.
        """
        # TODO fix summation
        """
        Fixing summation requires being able to slice of dimension(s) off a tensor which could then be used to create
        a tensor of ones which could be added to operation names.
        So first slicing needs to be implemented?
        Do we really need to slice that dimension or is it possible to just get tensor's shape?
        """
        try:
            order = list(range(len(self.opnames)))

            loc = self.operands.index(wrt)
            order[loc], order[-1] = order[-1], order[loc]

            # this is concatenation of two lists in np array
            operands_with_grad = list(np.array(self.operands + [grad])[order])

            opnames = list(np.array(self.opnames)[order])

            # add explicit Variables for implicitly summed out tensors
            from core.reshape import Shape
            wrt_shape = Shape(wrt)
            for i, letter in enumerate(self.opnames[loc]):
                if letter not in "".join(opnames[:-1]):
                    opnames.insert(0, letter)

                    dim = wrt_shape[i]
                    # we're running here dim()!!! that shouldn't be done?
                    var_to_insert = Variable(np.ones(dim()))
                    operands_with_grad.insert(0, var_to_insert)
            op_str = Einsum.to_einsum_string(opnames)

            return Einsum(op_str, *operands_with_grad[:-1])

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

    def f(self):
        val = self.node()
        return val * (val > 0)

    @module_wrapper
    def graph_df(self, wrt, grad):
        if self.node == wrt:
            return grad * self * Recipr(self)
        return 0


class Equal(Primitive):
    def __init__(self, first, second, name="Equal"):
        super().__init__([first, second], name)
        self.first, self.second = self.children

    def f(self):
        return np.equal(self.first(), self.second())

    def graph_df(self, wrt, grad):
        # is this correct?
        return 0


class Softmax(Module):
    """
    Softmax is a vector function: R^n -> R^n and taking its partial derivative is a bit tricky?
    We have a jacobian instead of a gradient?
    Interesting property, softmax is not invariant to scaling?

    """

    def __init__(self, node, ind, name="Softmax", expand_graph=True):
        super().__init__([node], name, expand_graph=expand_graph)
        self.node = self.children[0]
        self.ind = ind  # index of the element whose Softmax we want to compute
        self.out = self.init_graph()

    def graph(self):
        exp_ind = Exp(self.node[self.ind])
        exp_all = Exp(self.node)
        return exp_ind / Einsum("j->", exp_all)

    @module_wrapper
    def graph_df(self, wrt, grad):
        # gradient depends on every ind in self.node() ?
        if wrt == self.node:
            arr = []
            for dim in range(self.node().shape[-1]):
                if dim == self.ind:
                    val = self * (1 - self)
                else:
                    val = - self * Softmax(self.node, dim)
                arr.append(val)
            arr = np.array([op() for op in arr])
            return grad * Variable(arr)
        return 0


class Pow(Primitive):
    def __init__(self, first, second, name="Pow"):
        super().__init__([first, second], name)
        self.first = self.children[0]
        self.second = self.children[1]

    def f(self):
        f = self.first()
        s = self.second()

        return np.power(f, s)

    @module_wrapper
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

    def f(self):
        return np.log(self.node() + Primitive.epsilon)

    @module_wrapper
    def graph_df(self, wrt, grad):
        if self.node == wrt:
            return grad * Recipr(self.node)
        return 0


class Exp(Primitive):
    def __init__(self, node, name="Exp"):
        super().__init__([node], name)
        self.node = self.children[0]

    def f(self):
        return np.exp(self.node())

    @module_wrapper
    def graph_df(self, wrt, grad):
        if self.node == wrt:
            return grad * self
        return 0


class Tanh(Module):
    def __init__(self, node, name="Tanh", expand_graph=False):
        super().__init__([node], name, expand_graph=expand_graph)
        self.out = self.init_graph()

    def graph(self):
        node = self.children[0]
        return 2 * Sigmoid(2 * node) - 1


class Sigmoid(Primitive):
    def __init__(self, node, name="Sigmoid"):
        super().__init__([node], name=name)
        self.node = self.children[0]

    def f(self):
        return 1 / (1 + np.exp(-self.node()))

    @module_wrapper
    def graph_df(self, wrt, grad):
        if wrt == self.node:
            return grad * self * (1 - self)
        return 0


class SquaredDifference(Module):
    def __init__(self, first, second, name="Squared_diff", expand_graph=False):
        super().__init__([first, second], name=name, expand_graph=expand_graph)
        self.out = self.init_graph()

    def graph(self):
        first = self.children[0]
        second = self.children[1]

        diff = first - second
        return diff * diff


class TestRecursivelyComposite(Module):
    def __init__(self, node, count=1, name="Test_Composite", expand_graph=True):
        super().__init__([node], name=name, expand_graph=expand_graph)
        self.count = count
        self.out = self.init_graph()

    def graph(self):
        node = self.children[0]
        t = Variable(7, name="t")

        if self.count:
            return node * (1 + TestRecursivelyComposite(node,
                                                        count=self.count - 1,
                                                        expand_graph=self.expand_graph))
        else:
            return SquaredDifference(node, t, expand_graph=True)
