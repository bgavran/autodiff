import re
import numpy as np
import numbers
from automatic_differentiation.src.core.computational_graph import Node, Primitive, Variable

from functools import reduce
import string


def shape_from_elems(*elems):
    if len(elems) == 0:
        return 1,
    return np.broadcast(*[np.ones(elem.shape) for elem in elems]).shape


def reduce_grad(curr_grad, wrt):
    if curr_grad.shape == wrt.shape:
        return curr_grad
    curr_grad_letters = string.ascii_lowercase[:len(curr_grad.shape)]
    if len(wrt.shape) == 0:
        wrt_letters = ""
    else:
        wrt_letters = curr_grad_letters[-len(wrt.shape):]  # take last letters of curr_grad_letters

    # sum to leave just the wrt letters
    # but this wont work for more complex np broadcasting!
    # if some dimension length is equal to
    new_curr_grad = Einsum(str(curr_grad_letters) + "->" + str(wrt_letters), curr_grad)
    return new_curr_grad


class Add(Primitive):
    def __init__(self, *elems, name="Add"):
        if not elems:
            name = "0-" + name
        super().__init__(list(elems), name)
        self.shape = shape_from_elems(*self.children)

    def f(self):
        # Using python sum instead of np.sum because python converts types correctly
        return np.array(sum([elem() for elem in self.children]))

    def graph_df(self, wrt, curr_grad):
        # curr_grad will always be of shape of the shape of the "largest" variable
        # we need to sum across those other axes

        wrt_count = self.children.count(wrt)
        grad = curr_grad * Variable(wrt_count)
        return reduce_grad(grad, wrt)


class Mul(Primitive):
    fn = lambda x, y: x * y

    def __init__(self, *elems, name="Mul"):
        if not elems:
            name = "1-" + name
        super().__init__(list(elems), name)
        self.shape = shape_from_elems(*self.children)

    def f(self):
        # Mul broadcasts
        return reduce(Mul.fn, [child() for child in self.children], 1)

    def graph_df(self, wrt, curr_grad):
        # curr_grad will always be of shape of the shape of the "largest" variable ?
        # we need to sum across those other axes ?
        add_list = []
        for loc, child in enumerate(self.children):
            if child == wrt:
                add_list.append(Mul(*[ch for i, ch in enumerate(self.children) if loc != i]))

        grad = curr_grad * Add(*add_list)
        return reduce_grad(grad, wrt)


class Negate(Primitive):
    def __init__(self, node, name="Negate"):
        super().__init__([node], name)
        self.node = node
        self.shape = self.node.shape

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
        self.shape = self.node.shape

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
        self.shape = self.node.shape[::-1]

    def f(self):
        return np.transpose(self.node())

    def graph_df(self, wrt, curr_grad):
        if self.node == wrt:
            return Transpose(curr_grad)
        return 0


class Einsum(Primitive):
    def __init__(self, op_str, *operands, name="EinSum"):
        super().__init__(list(operands), name + " " + op_str)
        # TODO what if in the inputs there's two same variables?
        # TODO what if there's ellipsis in the op_str?
        self.op_str = op_str
        self.operands = self.children

        self.opnames = re.split(",|->", self.op_str)
        self.all_letters = "".join(set("".join(self.opnames[:-1])))
        # can also be "..." to an arbitrary shape tuple
        self.letter_to_dim = {}

        if len(self.operands) + 1 != len(self.opnames):
            raise ValueError("Number of operands doesn't match the einsum string!")

        for op, op_letters in zip(self.operands, self.opnames[:-1]):
            if len(op.shape) != 0 and len(op.shape) != len(op_letters) \
                    and "..." not in op_letters and op_letters != "":
                raise ValueError("Dimension of operand " + str(op) + " doesn't match the string! " +
                                 "Shape: " + str(op.shape) + " , string: '" + op_letters + "'")

            op_letters = op_letters[::-1]
            shp = op.shape[::-1]
            for i, lett in enumerate(Einsum.split_dots(op_letters)):
                try:
                    if len(lett) == 1:
                        dim = [shp[i]]  # what if shape is an empty tuple?
                    else:
                        dim = shp[i:]
                    if self.letter_to_dim.get(lett, dim) != dim:
                        raise ValueError("Inconsistent dimension names!")
                    self.letter_to_dim[lett] = dim
                except IndexError:
                    pass  # letters that we can't add are just dimension 1

        # problem with inferring shape from "..." string in Einsum!
        # TODO devise a way of inferring the shape of "..."
        self.shape = []
        for let in Einsum.split_dots(self.opnames[-1]):
            for l in self.letter_to_dim.get(let, [1]):
                self.shape.append(l)

    @staticmethod
    def split_dots(op_str):
        match_string = "\.{3}|\S"
        return re.findall(match_string, op_str)

    def f(self):
        arr = [op() for op in self.operands]

        for i, val in enumerate(arr):
            if isinstance(val, numbers.Number):
                shp = [l for let in Einsum.split_dots(self.opnames[i]) for l in self.letter_to_dim.get(let, [1])]
                arr[i] = np.broadcast_to(val, shp)

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

            # here we add explicit Variables for implicitly summed out tensors
            for i, letter in enumerate(Einsum.split_dots(self.opnames[loc])):
                if letter not in Einsum.split_dots("".join(opnames[:-1])):
                    opnames.insert(0, letter)

                    dim = wrt.shape[i]
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
        self.shape = self.node.shape

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

    """

    def __init__(self, node, name="Softmax"):
        super().__init__([node], name)
        self.node = self.children[0]
        self.shape = self.node.shape

    def f(self):
        """

        Subtracting the max of last axis from each element in softmax.
        Dividing the exp(node) by the sum of exp(node) for all nodes.
        Thes "one" variable is added so we can use softmax on tensors of arbitrarily high dimensions and sum back their
        last axis

        """
        val = self.node()
        shifted_exp = np.exp(val - np.expand_dims(np.max(val, axis=-1), axis=-1))

        # using my Einsum instead of numpy's since mine broadcasts them in a way that works well for autodiff
        shifted_exp_var = Variable(shifted_exp, name="shifted_exp")
        one_var = Variable(np.array([1]), name="1")
        last_axis_sum = Einsum("...j,o->...o", shifted_exp_var, one_var)()
        return shifted_exp / last_axis_sum

    def graph_df(self, wrt, curr_grad):
        # TODO higher order gradients don't work because Einsum grad can't be taken if ellipsis is used!
        if wrt == self.node:
            # matrix of the self outer product
            outer = Einsum("...i,...j->...ij", curr_grad * self, self)

            summ = reduce_grad(curr_grad, Variable(np.zeros(self.shape[-1])))
            ones_diag = Variable(np.eye(self.shape[-1]), "einsum_ones") # * summ
            # matrix where the only nonzero elements are the softmax vector on the diagonal
            # ij subscripts are both the same size, but np.einsum doesn't allow them with the same label
            kronecker_val = Einsum("ij,...j->...ij", ones_diag, self)

            a = Einsum("...ij->...j", kronecker_val - outer)
            return a
        return 0


class SoftmaxCEWithLogits(Primitive):
    def __init__(self, labels, logits, name="SoftmaxCEWithLogits"):
        super().__init__([labels, logits], name=name)
        self.labels = labels
        self.logits = logits

        self.shape = self.logits.shape[:-1]

    def f(self):
        labels_val = self.labels()
        logits_val = self.logits()
        labels_sum = np.sum(labels_val, axis=1)
        if not np.allclose(labels_sum, np.ones_like(labels_sum)):
            raise ValueError("Labels must be a valid probability distribution!")

        # calculating a numberically stable logsumpexp by shifting all the values
        maxx = np.expand_dims(np.max(logits_val, axis=-1), axis=-1)
        logsumexp = maxx + np.expand_dims(np.log(np.sum(np.exp(logits_val - maxx), axis=-1)), axis=-1)

        s = -np.sum(labels_val * logits_val - labels_val * logsumexp, axis=-1)
        return s

    def graph_df(self, wrt, curr_grad):
        if wrt == self.logits:
            return Einsum("...i,...ij->...ij", curr_grad, Softmax(self.logits) - self.labels)
        elif wrt == self.labels:
            return Variable(0)
        return 0


class SigmoidCEWithLogits(Primitive):
    def __init__(self, labels, logits, name="SigmoidCEWithLogits"):
        super().__init__([labels, logits], name)
        self.labels = labels
        self.logits = logits
        self.shape = self.logits.shape

    def f(self):
        z = self.labels()
        x = self.logits()
        return np.maximum(x, 0) - x * z + np.log(1 + np.exp(-abs(x)))

    def graph_df(self, wrt, curr_grad):
        if wrt == self.logits:
            return Einsum("...ij,...ij->...ij", curr_grad, Sigmoid(self.logits) - self.labels)
        return 0


class Pow(Primitive):
    def __init__(self, first, second, name="Pow"):
        super().__init__([first, second], name)
        self.first = self.children[0]
        self.second = self.children[1]
        self.shape = shape_from_elems(*self.children)

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
        self.shape = self.node.shape

    def f(self):
        return np.log(self.node() + Primitive.epsilon)

    def graph_df(self, wrt, curr_grad):
        if self.node == wrt:
            return curr_grad * Recipr(self.node)
        return 0


class Identity(Primitive):
    def __init__(self, node, name="Identity"):
        super().__init__([node], name)
        self.node = self.children[0]
        self.shape = self.node.shape

    def f(self):
        return self.node()

    def graph_df(self, wrt, curr_grad):
        if self.node == wrt:
            return curr_grad
        return 0


class Exp(Primitive):
    def __init__(self, node, name="Exp"):
        super().__init__([node], name)
        self.node = self.children[0]
        self.shape = self.node.shape

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
        self.shape = self.node.shape

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
        self.shape = 1,

    def f(self):
        return np.sqrt(sum([np.sum(np.square(node())) for node in self.nodes]))

    def graph_df(self, wrt, curr_grad):
        try:
            loc = self.nodes.index(wrt)
        except ValueError:
            return 0
        return curr_grad * self.nodes[loc] / self
