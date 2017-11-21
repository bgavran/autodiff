import numbers
import numpy as np
from automatic_differentiation.src.core.computational_graph import Primitive


class ReduceSumKeepDims(Primitive):
    def __init__(self, node, axes):
        super().__init__([node], name="ReduceSumKeepDims")
        self.node = self.children[0]
        self.axes = tuple(axes)
        self.shape = [1 if i in self.axes else shp for i, shp in enumerate(self.node.shape)]

    def _eval(self):
        return np.sum(self.node(), axis=self.axes, keepdims=True)

    def _partial_derivative(self, wrt, previous_grad):
        if self.node == wrt:
            return previous_grad * np.ones(self.node.shape)
        return 0


class Concat(Primitive):
    def __init__(self, a, b, axis=0):
        assert axis >= 0  # if axis is -1, how do we find out how many axes are there?
        super().__init__([a, b], name="Concat")
        self.a, self.b = self.children
        self.axis = axis

        self.shape = list(self.a.shape)
        self.shape[axis] += self.b.shape[axis]

    def _eval(self):
        a_val = self.a()
        b_val = self.b()
        return np.concatenate((a_val, b_val), axis=self.axis)

    def _partial_derivative(self, wrt, previous_grad):
        previous_grad = Reshape(previous_grad, self.shape)  # in case it's just a scalar Variable
        split = self.a.shape[self.axis]

        slice_val = [slice(None, None, None) for _ in range(self.axis + 1)]
        if wrt == self.a:
            slice_val[self.axis] = slice(None, split, None)
            return previous_grad[slice_val]
        elif wrt == self.b:
            # need to broadcast grad to shape of self?
            slice_val[self.axis] = slice(split, None, None)
            return previous_grad[slice_val]
        return 0


class Reshape(Primitive):
    def __init__(self, node, shape, name="Reshape"):
        super().__init__([node], name)
        self.node = self.children[0]
        self.shape = self.infer_shape(shape)  # because some some dimension might be -1

    def _eval(self):
        node_val = self.node()
        if isinstance(node_val, numbers.Number):
            return np.broadcast_to(node_val, self.shape)
        return np.reshape(node_val, self.shape)

    def _partial_derivative(self, wrt, previous_grad):
        if self.node == wrt:
            return Reshape(previous_grad, self.node.shape)
        return 0

    def infer_shape(self, shape):
        if isinstance(shape, numbers.Number):
            return shape
        if -1 in shape:
            shape = list(shape)
            for i in range(len(shape)):
                if shape[i] == -1:
                    shape[i] = int(-np.prod(self.node.shape) / np.prod(shape))
        return shape


# TODO does slicing really work as it should?! Higher order gradients seem wrong?

class Slice(Primitive):
    def __init__(self, node, slice_val, name="Slice"):
        if name is None:
            name = str(slice_val)
        super().__init__([node], name)
        self.node = self.children[0]
        self.slice_val = slice_val

        self.shape = np.zeros(self.node.shape)[self.slice_val].shape

    def _eval(self):
        val = self.node()
        return val[self.slice_val]

    def _partial_derivative(self, wrt, previous_grad):
        # TODO does it work for higher order derivatives, since previous_grad is evaluated here?
        if self.node == wrt:
            grad = np.zeros(wrt.shape)
            grad[self.slice_val] = previous_grad()
            return grad
        return 0


class Pad(Primitive):
    def __init__(self, node, pad_width, constant_values, name="Slice"):
        """

        :param node:
        :param pad_width:  different than pad_width arg in np.pad, this one pads up to the length provided
        :param constant_values:
        :param name:
        """
        super().__init__([node], name)
        self.node = self.children[0]
        self.pad_width = pad_width
        self.constant_values = constant_values

        self.shape = np.pad(np.ones(self.node.shape),
                            self.pad_width,
                            mode="constant",
                            constant_values=self.constant_values).shape

    def _eval(self):
        val = self.node()
        return np.pad(val,
                      self.pad_width,
                      mode="constant",
                      constant_values=self.constant_values)

    def _partial_derivative(self, wrt, previous_grad):
        if self.node == wrt:
            slice_val = [slice(pad[0], shp - pad[1]) for pad,shp in zip(self.pad_width, self.shape)]
            return previous_grad[slice_val]
        return 0

#
# class AsStrided(Primitive):
#     def __init__(self, node, shape, strides):
#         super().__init__([node], name="AsStrided")
#         self.node = self.children[0]
#         self.shape = shape
#         self.strides = strides
#
#     def _eval(self):
#         return as_strided(self.node(), shape=self.shape, strides=self.strides)
#
#     def partial_derivative(self, wrt, previous_grad):
#         if self.node == wrt:  # TODO strides need to be known before graph evaluation?
#             seq = np.arange(np.prod(self.node.shape)).reshape(*self.node.shape)
#             as_stride = as_strided(seq, shape=self.shape, strides=self.strides)
#             it = np.nditer(seq, flags=['multi_index'])
#
#             res = np.zeros(self.node.shape)
#             while not it.finished:
#                 ind = it.multi_index
#                 res[ind] = ((as_stride == seq[ind]) * previous_grad()).sum()
#                 it.iternext()
#             return Variable(res)
#         return 0
