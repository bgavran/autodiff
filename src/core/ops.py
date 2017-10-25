import re
import numpy as np
import numbers
from automatic_differentiation.src.core.computational_graph import Node, Primitive, Variable

from functools import reduce


class Add(Primitive):
    def __init__(self, *elems, name="Add"):
        if not elems:
            name = "0-" + name
        super().__init__(list(elems), name)

    def f(self):
        # Using python sum instead of np.sum because python converts types correctly
        return sum([elem() for elem in self.children])

    def graph_df(self, wrt, curr_grad):
        wrt_count = self.children.count(wrt)
        return curr_grad * Variable(wrt_count)


class Identity(Primitive):
    def __init__(self, node, name="Identity"):
        super().__init__([node], name)
        self.node = self.children[0]

    def f(self):
        return self.node.f()

    def graph_df(self, wrt, curr_grad):
        if self.node == wrt:
            return curr_grad * self.node.graph_df(wrt, curr_grad)
        return 0


class Mul(Primitive):
    fn = lambda x, y: x * y

    def __init__(self, *elems, name="Mul"):
        if not elems:
            name = "1-" + name
        super().__init__(list(elems), name)

    def f(self):
        return reduce(Mul.fn, [child() for child in self.children], 1)

    def graph_df(self, wrt, curr_grad):
        add_list = []
        for loc, child in enumerate(self.children):
            if child == wrt:
                add_list.append(Mul(*[ch for i, ch in enumerate(self.children) if loc != i]))
        return curr_grad * Add(*add_list)


class Negate(Primitive):
    def __init__(self, node, name="Negate"):
        super().__init__([node], name)
        self.node = node

    def f(self):
        return -self.node()

    def graph_df(self, wrt, curr_grad):
        if self.node == wrt:
            return -curr_grad
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
        return 1 / (self.node() + Primitive.epsilon)

    def graph_df(self, wrt, curr_grad):
        if self.node == wrt:
            return - curr_grad * self * self
        return 0


class Transpose(Primitive):
    def __init__(self, node, name="Transpose"):
        super().__init__([node], name)
        self.node = self.children[0]

    def f(self):
        return np.transpose(self.node())

    def graph_df(self, wrt, curr_grad):
        if self.node == wrt:
            return Transpose(curr_grad)
        return 0


class MatMul(Primitive):  # this is actually a Module
    def __init__(self, a, b, name="MatMul"):
        super().__init__([a, b], name)
        self.op = Einsum("ij,jk->ik", self.children[0], self.children[1])

    def f(self):
        return self.op()

    def graph_df(self, wrt, curr_grad):
        return self.op.graph_df(wrt, curr_grad)


class Einsum(Primitive):
    def __init__(self, op_str, *operands, name="EinSum"):
        super().__init__(list(operands), name + " " + op_str)
        # TODO what if in the inputs there's two same variables?
        # TODO what if there's ellipsis in the op_str?
        self.op_str = op_str
        self.operands = self.children

        self.opnames = re.split(",|->", self.op_str)
        self.all_letters = "".join(set("".join(self.opnames[:-1])))

        if len(self.operands) + 1 != len(self.opnames):
            raise ValueError("Number of operands doesn't match the einsum string!")

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
                if len(shape) != len(letters) and "..." not in letters:  # ellipsis check needs to be solved properly!
                    raise ValueError("Dimension of operand " + str(i + 1) + " doesn't match the string! " +
                                     "Shape: " + str(shape) + " , string: '" + letters + "'")
                for letter, dim in zip(letters, shape):
                    letter_to_dim[letter] = dim

        for i, val in enumerate(arr):
            if isinstance(val, numbers.Number):
                shape_in_letters = self.opnames[i]
                shape_in_dims = [letter_to_dim.get(letter, 1) for letter in shape_in_letters]
                arr[i] = np.broadcast_to(val, shape_in_dims)

        return np.einsum(self.op_str, *arr)

    def graph_df(self, wrt, curr_grad):
        """
        Usual einsum operation looks something like this c = einsum("ij,kj->ik", a, b)
        Gradient w.r.t. the first parameter just changes the op to look like this: df = einsum("ik,kj->ij", c, b).
        It basically just switches the output with one of the inputs.

        For tensors that have some of their dimensions implicitly summed, a new tensor of ones is explicitly added
        """
        if "..." in self.op_str:
            raise NotImplementedError("Grad of Einsum that uses ellipsis is not implemented yet!")
        else:
            order = list(range(len(self.opnames)))

            try:
                loc = self.operands.index(wrt)
            except ValueError:
                return 0
            order[loc], order[-1] = order[-1], order[loc]

            # this is concatenation of two lists in np array and then their reorder
            operands_with_grad = list(np.array(self.operands + [curr_grad])[order])

            opnames = list(np.array(self.opnames)[order])

            # we add here explicit Variables for implicitly summed out tensors
            from automatic_differentiation.src.core.reshape import Shape

            wrt_shape = Shape(wrt)

            for i, letter in enumerate(self.opnames[loc]):
                if letter not in "".join(opnames[:-1]):
                    opnames.insert(0, letter)

                    dim = wrt_shape[i]()
                    var_to_insert = Variable(np.ones(dim), name="np.ones(" + str(dim) + ")")
                    operands_with_grad.insert(0, var_to_insert)
            op_str = Einsum.to_einsum_string(opnames)

            return Einsum(op_str, *operands_with_grad[:-1])

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

    def graph_df(self, wrt, curr_grad):
        if self.node == wrt:
            return curr_grad * self * Recipr(self)
        return 0


class Softmax(Primitive):
    """
    Softmax is a vector function: R^n -> R^n and taking its partial derivative w.r.t. input is a Jacobian matrix.
    But it can be done for batches also?

    """

    def __init__(self, node, name="Softmax"):
        super().__init__([node], name)
        self.node = self.children[0]

    def f(self):
        """

        Subtracting the max of last axis from each element in softmax.
        Dividing the exp(node) by the sum of exp(node) for all nodes.
        Thes "one" variable is added so we can use softmax on tensors of arbitrarily high dimensions and sum back their
        last axis

        """
        val = self.node()
        shifted_exp = np.exp(val - np.expand_dims(np.max(val, axis=-1), axis=-1))
        one = np.array([1])

        # using my Einsum instead of numpy's since mine broadcasts them in a way that works well for autodiff
        last_axis_sum = Einsum("...j,o->...o", Variable(shifted_exp, name="shifted_exp"), Variable(one, name="1"))()
        return shifted_exp / last_axis_sum

    def graph_df(self, wrt, curr_grad):
        # TODO higher order gradients don't work because Einsum grad can't be taken if ellipsis is used!
        if wrt == self.node:
            # matrix of the self outer product
            outer = Einsum("...i,...j->...ij", self, self)

            val_ones = np.eye(self().shape[-1])
            ones_diag = Variable(val_ones, "einsum_ones")
            # matrix where the only nonzero elements are the softmax vector on the diagonal
            kronecker_val = Einsum("...ij,...j->...ij", ones_diag, self)

            return curr_grad * Einsum("...ij->...j", outer - kronecker_val)
        return 0


class SoftmaxCEWithLogits(Primitive):
    def __init__(self, labels, logits, name="SoftmaxCEWithLogits"):
        super().__init__([labels, logits], name=name)
        self.labels = labels
        self.logits = logits

    def f(self):
        labels_val = self.labels()
        logits_val = self.logits()

        # calculating a numberically stable logsumpexp by shifting all the values
        maxx = np.expand_dims(np.max(logits_val, axis=-1), axis=-1)
        logsumexp = maxx + np.expand_dims(np.log(np.sum(np.exp(logits_val - maxx), axis=-1)), axis=-1)

        s = -np.sum(labels_val * logits_val - labels_val * logsumexp, axis=-1)
        return s

    def graph_df(self, wrt, curr_grad):
        if wrt == self.logits:
            # TODO missing grad here!
            return Softmax(self.logits) - self.labels
        elif wrt == self.labels:
            return Variable(0)
        return 0


class Pow(Primitive):
    def __init__(self, first, second, name="Pow"):
        super().__init__([first, second], name)
        self.first = self.children[0]
        self.second = self.children[1]

    def f(self):
        return np.power(self.first(), self.second())

    def graph_df(self, wrt, curr_grad):
        if self.first == self.second == wrt:
            return curr_grad * self * (Log(self.first) + 1)
        elif self.first == wrt:
            return curr_grad * self.second * Pow(self.first, self.second - 1)
        elif self.second == wrt:
            return curr_grad * Log(self.first) * self
        return 0


class Log(Primitive):
    def __init__(self, node, name="Log"):
        super().__init__([node], name)
        self.node = self.children[0]

    def f(self):
        return np.log(self.node() + Primitive.epsilon)

    def graph_df(self, wrt, curr_grad):
        if self.node == wrt:
            return curr_grad * Recipr(self.node)
        return 0


class Exp(Primitive):
    def __init__(self, node, name="Exp"):
        super().__init__([node], name)
        self.node = self.children[0]

    def f(self):
        return np.exp(self.node())

    def graph_df(self, wrt, curr_grad):
        if self.node == wrt:
            return curr_grad * self
        return 0


class Sigmoid(Primitive):
    def __init__(self, node, name="Sigmoid"):
        super().__init__([node], name=name)
        self.node = self.children[0]

    def f(self):
        return 1 / (1 + np.exp(-self.node()))

    def graph_df(self, wrt, curr_grad):
        if wrt == self.node:
            return curr_grad * self * (1 - self)
        return 0


class FrobeniusNorm(Primitive):
    def __init__(self, *nodes, name="Frobenius Norm"):
        super().__init__(list(nodes), name=name)
        self.nodes = nodes

    def f(self):
        return np.sqrt(sum([np.sum(np.square(node())) for node in self.nodes]))

    def graph_df(self, wrt, curr_grad):
        try:
            loc = self.nodes.index(wrt)
        except ValueError:
            return 0
        return self.nodes[loc] / self
