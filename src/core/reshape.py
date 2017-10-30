from automatic_differentiation.src.core.computational_graph import *


class Concat(Primitive):
    def __init__(self, a, b, axis=0):
        assert axis >= 0  # if axis is -1, how do we find out how many axes are there?
        super().__init__([a, b], name="Concat")
        self.a, self.b = self.children
        self.axis = axis

        self.shape = list(self.a.shape)
        self.shape[axis] += self.b.shape[axis]

    def f(self):
        a_val = self.a()
        b_val = self.b()
        return np.concatenate((a_val, b_val), axis=self.axis)

    def graph_df(self, wrt, curr_grad):
        curr_grad = Reshape(curr_grad, self.shape)  # in case it's just a scalar Variable
        split = self.a.shape[self.axis]

        slice_val = [slice(None, None, None) for _ in range(self.axis + 1)]
        if wrt == self.a:
            slice_val[self.axis] = slice(None, split, None)
            return curr_grad[slice_val]
        elif wrt == self.b:
            # need to broadcast grad to shape of self?
            slice_val[self.axis] = slice(split, None, None)
            return curr_grad[slice_val]
        return 0


class Reshape(Primitive):
    def __init__(self, node, shape, name="Reshape"):
        super().__init__([node], name)
        self.node = self.children[0]
        self.shape = self.infer_shape(shape)  # because some some dimension might be -1

    def f(self):
        node_val = self.node()
        if isinstance(node_val, numbers.Number):
            return np.broadcast_to(node_val, self.shape)
        return np.reshape(node_val, self.shape)

    def graph_df(self, wrt, curr_grad):
        if self.node == wrt:
            return Reshape(curr_grad, self.node.shape)
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
    # TODO slicing with the double dot operator?
    def __init__(self, node, slice_val, name="Slice"):
        # stop value must be negative and step must be None (constraint of current implementation)

        # cond = lambda slc: (slc.stop is None or slc.stop < 0) and slc.step is None
        # if isinstance(slice_val, tuple) or isinstance(slice_val, list):
        # for sl in slice_val:
        #     assert cond(sl)
        # else:
        # assert cond(slice_val)
        if name is None:
            name = str(slice_val)
        super().__init__([node], name)
        self.node = self.children[0]
        self.slice_val = slice_val

        self.shape = self().shape  # TODO fix hack

    def f(self):
        val = self.node()
        if isinstance(val, numbers.Number):
            return val
        return val[self.slice_val]

    def graph_df(self, wrt, curr_grad):
        # TODO how to do this in a simple way? Also needs to support reversal of tensor ([::-1])
        if self.node == wrt:
            curr_grad = Reshape(curr_grad, self.shape)  # in case it's just a scalar Variable

            if isinstance(self.slice_val, list) or isinstance(self.slice_val, tuple):
                pad_val = [[0 if sl.start is None else sl.start,
                            0 if sl.stop is None else -sl.stop] for sl in self.slice_val]
                return Pad(curr_grad, pad_val, constant_values=[0 for _ in self.slice_val])
            else:
                return curr_grad
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

        self.shape = self.node.shape  # TODO fix this line!

    def f(self):
        val = self.node()
        return np.pad(val, self.pad_width, mode="constant", constant_values=self.constant_values)

    def graph_df(self, wrt, curr_grad):
        if self.node == wrt:
            # problem: pad[1] is always positive here?
            # it seems impossible to guarantee that slice inputs will be negative?
            # TODO there seems to be a need for a more fundamental solution to this!
            slice_val = [slice(pad[0], pad[1]) for pad in self.pad_width]
            return curr_grad[slice_val]
        return 0
