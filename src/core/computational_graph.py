import os
import numbers
import collections
import numpy as np
from contextlib import contextmanager


class Node:
    id = 0
    context = []

    def __init__(self, name):
        self.id = Node.id
        Node.id += 1

        self.context = Node.context.copy()
        if name == "":
            self.name = os.path.join("id_" + str(self.id))
        else:
            self.name = os.path.join(name)

    def __str__(self):
        return "/".join(self.context + [self.name])

    def __add__(self, other):
        from core.ops import Add
        return Add(self, other)

    def __radd__(self, other):
        return self.__add__(other)

    def __neg__(self):
        from core.ops import Negate
        return Negate(self)

    def __sub__(self, other):
        return self.__add__(other.__neg__())

    def __rsub__(self, other):
        return self.__neg__().__add__(other)

    def __mul__(self, other):
        from core.ops import Mul
        return Mul(self, other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __matmul__(self, other):
        from core.ops import MatMul
        return MatMul(self, other)

    def __rmatmul__(self, other):
        from core.ops import MatMul
        return MatMul(other, self)

    def __imatmul__(self, other):
        return self.__matmul__(other)

    def __truediv__(self, other):
        from core.ops import Recipr
        return self.__mul__(Recipr(other))

    def __rtruediv__(self, other):
        from core.ops import Recipr
        return Recipr(self).__mul__(other)

    @staticmethod
    def _constant_wrapper(init_function):
        """
        Decorator for the Node class which wraps any normal number into a Constant
        """

        def wrap_all_children(self, children, name):
            for i, child in enumerate(children):
                if isinstance(child, numbers.Number):
                    children[i] = Constant(child)
                assert isinstance(children[i], Node)
            return init_function(self, children, name)

        return wrap_all_children

    @staticmethod
    @contextmanager
    def add_context(ctx):
        Node.context.append(ctx)
        try:
            yield
        finally:
            Node.context.pop()

    def get_node_for_graph(self):
        return self

    def add_node_with_context(self, digraph, ctx):
        """
        Add just the node (not the connections, not the children) to the respective subgraph
        :param digraph:
        :param ctx:
        :return:
        """
        if len(ctx):
            with digraph.subgraph(name="cluster" + str(ctx[0])) as subgraph:
                subgraph.attr(color="blue")
                subgraph.attr(label=ctx[0])

                self.add_node_with_context(subgraph, ctx[1:])
        else:
            digraph.add_node(self)


class Operation(Node):
    epsilon = 1e-12

    @Node._constant_wrapper
    def __init__(self, children, name=""):
        super().__init__(name)
        self.children = children

    def __iter__(self):
        yield self
        for child in self.children:
            yield from child

    def topo_sort(self):
        # iterator with duplicates
        ll = reversed(list(self))

        # returns list without duplicates
        return list(collections.OrderedDict.fromkeys(ll))

    def all_nodes(self):
        return set(self)

    def find_node_by_id(self, node_id):
        for node in self.all_nodes():
            if node.id == node_id:
                return node
        raise ValueError("Node with id", node_id, "doesn't exist!")

    def eval(self, input_dict):
        raise NotImplementedError()

    def graph_df(self, wrt, grad=None):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.eval(*args, **kwargs)

    def add_node_subgraph_to_plot_graph(self, digraph):
        if str(self.id) not in digraph.set_of_added_nodes:
            # Add node
            self.add_node_with_context(digraph, self.context)

            # Add connections to children
            for child in self.children:
                digraph.add_edge(child, self)

            # Make each of the children do the same
            for child in self.children:
                child.add_node_subgraph_to_plot_graph(digraph)


class Constant(Operation):
    def __init__(self, value, name=None):
        assert isinstance(value, np.ndarray) or isinstance(value, numbers.Number)
        if name is None:
            name = str(value)
        super().__init__([], name)
        self.value = value

    def eval(self, input_dict):
        return self.value

    def graph_df(self, wrt="", grad=None):
        return 0


class Variable(Operation):
    def __init__(self, name=""):
        """

        Right now, all variables are placeholders?
        All variables need their values to be fed in.
        This class needs to be separated into Variables which have their value? Something like a constant?
        Except it changes?

        :param name:
        """
        super().__init__([], name)

    def eval(self, input_dict):
        try:
            return input_dict[self]
        except KeyError:
            raise KeyError("Must input value for variable: " + str(self))

    def graph_df(self, wrt, grad=None):
        if self == wrt:
            return grad
        return 0


class Identity(Operation):
    def __init__(self, node, name="Identity"):
        super().__init__([node], name)
        self.node = self.children[0]

    def eval(self, input_dict):
        return self.node.eval(input_dict)

    def graph_df(self, wrt, grad=None):
        return self.node.graph_df(wrt, grad)


class CompositeWrapper:
    @staticmethod
    def from_function(fn):
        def wrap_in_composite(*fn_args, expand_when_graphed=True):
            op = CompositeOperation(children=fn_args,
                                    name=fn.__name__,
                                    expand_when_graphed=expand_when_graphed)

            op.graph = lambda: fn(*op.children)
            op.out = op.init_graph()

            return op

        return wrap_in_composite

    @staticmethod
    def from_graph_df(fn):
        def wrap_in_composite(instance, wrt, grad=None, expand_when_graphed=False):
            # TODO what are the children of this CompOp?
            """
            Should the instance be a child?
            Because we could then go into an infinite loop by trying to diff the instance again?

            """
            name = "Gradient graph of " + instance.name + " "
            # children = [grad]
            # children = [grad, instance]
            children = [grad] + list(set(instance.children))
            # children = [grad] + [instance] + list(set(instance.children))

            op = CompositeOperation(children,
                                    name=name,
                                    expand_when_graphed=expand_when_graphed)

            op.graph = lambda: fn(instance, wrt, grad)
            op.out = op.init_graph()

            return op

        return wrap_in_composite


class CompositeOperation(Operation):
    def __init__(self, children, name="", expand_when_graphed=True):
        """

        :param children:
        :param name: 
        """
        super().__init__(children, name)
        self.expand_when_graphed = expand_when_graphed
        self._out = None

    def init_graph(self):
        ctx = self.name + " " + str(self.id)
        with Node.add_context(ctx):
            out = self.graph()
        assert out is not None

        return out

    @property
    def out(self):
        return self._out

    @out.setter
    def out(self, val):
        if self._out is None:
            self._out = val
        else:
            raise ValueError("_out variable for instance " + self.name + " is already set!")

    def get_node_for_graph(self):
        if self.expand_when_graphed:
            return self.out.get_node_for_graph()
        else:
            return self

    def add_node_subgraph_to_plot_graph(self, digraph):
        if self.expand_when_graphed:
            self.get_node_for_graph().add_node_subgraph_to_plot_graph(digraph)
        else:
            Operation.add_node_subgraph_to_plot_graph(self, digraph)

    def eval(self, input_dict):
        return self.out.eval(input_dict)

    @CompositeWrapper.from_graph_df
    def graph_df(self, wrt, grad=None):
        gr = Grad(self.out,
                  wrt=wrt,
                  initial_grad=grad,
                  name=self.name)

        return gr

    def graph(self):
        """
        In this method all the operations on the children need to be defined and the result returned.

        :return:
        """
        raise NotImplementedError()


class Add(Operation):
    def __init__(self, *elems, name="Add"):
        if not elems:
            name = "0-Add"
        super().__init__(list(elems), name)

    def eval(self, input_dict):
        arr = [elem(input_dict) for elem in self.children]
        # Using python sum instead of np.sum because python converts types correctly
        return sum(arr)

    @CompositeWrapper.from_graph_df
    def graph_df(self, wrt, grad=None):
        wrt_count = self.children.count(wrt)
        return grad * Constant(wrt_count)


# TODO 2nd gradients problem!
"""

Graph_df became a CompositeOperation now. 
CompositeOperation needs to have its children known on the time of creation, 
but an arbitrary graph_df can have branching inside it and whatnot.

It means that it's not possible to know graph_df's children at creation time.
Is CompositeOperation the right pattern here? 

Problem is that CompositeOperation graph() function depends on the children passed to it at 
initialization time.

But what exactly is the problem with passing all children of an Op to the CompositeOperation graph_df?

--
Well, if we need to pass all the children, we might as well pass the instance as well, since sometimes the
grad depends on the fn itself (Exp() for example)
But then we get into an infinite loop?

"""


class Grad(CompositeOperation):
    def __init__(self, node, wrt, initial_grad=None, name="", expand_when_graphed=True):
        super().__init__([node],
                         name=name + " grad w.r.t. " + str(wrt),
                         expand_when_graphed=expand_when_graphed)
        self.wrt = wrt
        self.initial_grad = Constant(1, name=node.name + "_grad") if initial_grad is None else initial_grad
        self.out = self.init_graph()

    def graph(self):
        out_node = self.children[0]
        nodes = out_node.topo_sort()

        if self.wrt not in nodes:
            # raise ValueError("Node with the name \"" + str(self.wrt) + "\" is not in the graph!")
            return Constant(0, name=str(self.wrt) + "_zero")

        dct = collections.defaultdict(list)
        dct[out_node.id].append(self.initial_grad)

        for node in reversed(nodes):
            dct[node.id] = Add(*dct[node.id], name=node.name + "_grad_sum")

            for child in set(node.children):
                app = node.graph_df(child, grad=dct[node.id])
                dct[child.id].append(app)

        return dct[self.wrt.id]
