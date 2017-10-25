from automatic_differentiation.src.core.computational_graph import *


class Concat(Primitive):
    def __init__(self, a, b, axis=0):
        assert axis >= 0  # if axis is -1, how do we find out how many axes are there?
        super().__init__([a, b], name="Concat")
        self.a = a
        self.b = b
        self.axis = axis

    def f(self):
        a_val = self.a()
        b_val = self.b()
        return np.concatenate((a_val, b_val), axis=self.axis)

    def graph_df(self, wrt, curr_grad):
        curr_grad = Reshape(curr_grad, Shape(self))  # in case it's just a scalar Variable
        split = self.a().shape[self.axis]

        slice_val = [slice(None, None, None) for _ in range(self.axis + 1)]
        if wrt == self.a:
            slice_val[self.axis] = slice(None, split, None)
            return curr_grad[slice_val]
        elif wrt == self.b:
            # need to broadcast grad to shape of self?
            slice_val[self.axis] = slice(split, None, None)
            return curr_grad[slice_val]
        return 0


class Shape(Primitive):
    def __init__(self, node=None, from_tuple=None, name="Shape"):
        # this seems to complex compared to the rest of the code?
        if node is None and from_tuple is None:
            raise ValueError("Must pass either a node or a shape tuple!")
        elif node is not None and from_tuple is not None:
            raise ValueError("Must pass just one of the arguments!")

        if node is not None:
            super().__init__([node], name)
            self.node = self.children[0]
        else:
            super().__init__([], name)
            self.shape = from_tuple

    def f(self):
        if hasattr(self, "node"):
            return np.array(self.node()).shape
        else:
            return self.shape

    def graph_df(self, wrt, curr_grad):
        # Should it be this way?
        return 0


class Reshape(Primitive):
    def __init__(self, node, shape, name="Reshape"):
        super().__init__([node], name)
        self.node = self.children[0]
        self.shape = shape  # shape is also a Node

    def f(self):
        node_val = self.node()
        like_node_shape = self.shape()
        if isinstance(node_val, numbers.Number):
            return np.broadcast_to(node_val, like_node_shape)
        return np.reshape(node_val, like_node_shape)

    def graph_df(self, wrt, curr_grad):
        if self.node == wrt:
            return Reshape(curr_grad, Shape(self.node))
        return 0


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

    def f(self):
        val = self.node()
        if isinstance(val, numbers.Number):
            return val
        return val[self.slice_val]

    def graph_df(self, wrt, curr_grad):
        # TODO how to do this in a simple way? Also needs to support reversal of tensor ([::-1])
        if self.node == wrt:
            curr_grad = Reshape(curr_grad, Shape(self))  # in case it's just a scalar Variable

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
