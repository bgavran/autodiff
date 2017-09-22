import os
import numbers
import collections
import numpy as np
from contextlib import contextmanager


class Context:
    def __init__(self, root_node):
        self.root_node = root_node
        self.inner_id = 0
        self.has_subnode_in_graph = False
        self.topo_sort = []

    def add_to_context(self, node):
        self.inner_id += 1
        self.topo_sort.append(node)
        return self.inner_id - 1

    def __str__(self):
        return str(self.root_node)


def _constant_wrapper(init_function):
    """
    Decorator for the Node class which wraps any normal number into a Constant
    """

    def wrap_all_children(self, children, name):
        for i, child in enumerate(children):
            if isinstance(child, numbers.Number):
                children[i] = Variable(child)
            if not isinstance(children[i], Node):
                raise ValueError("Input to op '" + name + "' is a " + str(type(children[i])) + "!!!")
        return init_function(self, children, name)

    return wrap_all_children


class Node:
    context_list = []

    @_constant_wrapper
    def __init__(self, children, name):
        self.children = children
        if len(Node.context_list) == 0:
            Node.context_list.append(Context(""))
        self.id = Node.context_list[-1].add_to_context(self)

        self.in_graph = False
        self.context_list = Node.context_list.copy()
        if name == "":
            self.name = os.path.join("id_" + str(self.id))
        else:
            self.name = os.path.join(name)

    def graph_name(self):
        return "/".join([str(ctx) for ctx in self.context_list]) + " " + str(self.id)

    def __str__(self):
        return self.name + " " + str(self.id)

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

    def get_node_for_graph(self):
        return self

    @staticmethod
    @contextmanager
    def add_context(ctx):
        Node.context_list.append(ctx)
        try:
            yield
        finally:
            del Node.context_list[-1]

    def apply_on_self_and_children(self, fn, *args, **kwargs):
        fn(self, *args, **kwargs)
        for child in set(self.children):
            Node.apply_on_self_and_children(child, fn, *args, **kwargs)

    @contextmanager
    def mark_node_in_graph(self):
        def mark_node(node, val):
            node.context_list[-1].has_subnode_in_graph = val
            node.in_graph = val

        self.apply_on_self_and_children(lambda node: mark_node(node, True))
        try:
            yield
        finally:
            self.apply_on_self_and_children(lambda node: mark_node(node, False))

    def reverse_topo_sort(self):
        # TODO document this method
        reverse_sorted_nodes = []
        for node in reversed(self.context_list[-1].topo_sort):
            if node.in_graph:
                reverse_sorted_nodes.append(node)
            elif isinstance(node, CompositeOperation) and node.context.has_subnode_in_graph:
                app = node.out.reverse_topo_sort()
                reverse_sorted_nodes.extend(app)

        return reverse_sorted_nodes

    def __iter__(self):
        yield self
        for child in set(self.children):
            yield from child

    def find_nodes(self, with_context=True):
        """
        Recursively traces back throughout the children of nodes (starting from this one).
        The elements of the list depend on with_context variable.
        If with_context=True:
            returns nodes with the same context_list
        else:
            returns just the top nodes that have different context_list (not their children)

        :return: a list of nodes
        """
        # TODO what's the complexity of this?

        ll = []
        if with_context:
            ll.append(self)

        for child in self.children:
            if child.context_list == self.context_list:
                ll.extend(child.find_nodes(with_context))
            else:
                if not with_context:
                    ll.append(child)
        return list(set(ll))

    def get_ctx_node(self):
        for ctx in self.context_list:
            if not ctx.root_node == "" and not ctx.root_node.graph_expand:
                return ctx.root_node
        return self

    def add_node_subgraph_to_plot_graph(self, digraph):
        if self.graph_name() not in digraph.set_of_added_nodes:
            digraph.add_node_with_context(self, self.context_list)

            # Add connections to children
            for child in self.children:
                digraph.add_edge(child.get_ctx_node(), self)

            # Make each of the children do the same
            for child in self.children:
                # If the child is part of a CompOp which is not expanded, add the CompOp to subgraph
                child.get_ctx_node().add_node_subgraph_to_plot_graph(digraph)


class Primitive(Node):
    epsilon = 1e-12

    def __init__(self, children, name=""):
        super().__init__(children, name)

    def eval(self):
        raise NotImplementedError()

    def graph_df(self, wrt, grad):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.eval()


class Variable(Primitive):
    def __init__(self, value, name=None):
        assert isinstance(value, np.ndarray) or isinstance(value, numbers.Number)
        if name is None:
            name = str(value)
        super().__init__([], name)
        self.value = value

    def eval(self):
        return self.value

    def graph_df(self, wrt, grad):
        if self == wrt:
            return grad
        return 0


class CompositeWrapper:
    @staticmethod
    def from_graph_df(fn):
        def wrap_in_composite(instance, wrt, grad, graph_expand=True):
            # maybe subclassing was a better idea here?

            # children = [grad] + list(set(instance.children))
            # we pass empty list of children here because they will be found anyway?
            name = "Gradient graph of " + instance.name + " "
            op = CompositeOperation(children=[], name=name, graph_expand=graph_expand)

            op.graph = lambda: fn(instance, wrt, grad)
            op.out = op.init_graph()

            return op

        return wrap_in_composite

    @staticmethod
    def from_function(fn):
        def wrap_in_composite(*fn_args, graph_expand=False):
            # we also don't need children here?
            op = CompositeOperation(children=fn_args, name=fn.__name__, graph_expand=graph_expand)

            # what if the function had depended on some variables outside of its scope? Do we allow for that?

            op.graph = lambda: fn(*op.children)
            op.out = op.init_graph()

            return op

        return wrap_in_composite


class CompositeOperation(Primitive):
    def __init__(self, children, name="", graph_expand=False):
        """

        :param children:
        :param name:
        """
        super().__init__(children, name)
        self.graph_expand = graph_expand
        self.context = Context(self)
        self._out = None

    def init_graph(self):
        with Node.add_context(self.context):
            out = self.graph()
        assert out is not None

        # this is needed only for Grad. Can it be a bit cleaner?
        self.children = out.find_nodes(with_context=False)

        # somehow simplify graphs here?

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
        if self.graph_expand:
            return self.out.get_node_for_graph()
        else:
            return self

    def add_node_subgraph_to_plot_graph(self, digraph):
        if self.graph_expand:
            self.out.add_node_subgraph_to_plot_graph(digraph)
        else:
            Primitive.add_node_subgraph_to_plot_graph(self, digraph)

    def eval(self):
        return self.out.eval()

    # @CompositeWrapper.from_graph_df
    def graph_df(self, wrt, grad):
        return Grad(self.out, wrt=wrt, initial_grad=grad, name=self.name, graph_expand=False)

    def graph(self):
        """
        In this method all the operations on the children need to be defined and the result returned.

        :return:
        """
        raise NotImplementedError()


class Add(Primitive):
    def __init__(self, *elems, name="Add"):
        if not elems:
            name = "0-" + name
        super().__init__(list(elems), name)

    def eval(self):
        # Using python sum instead of np.sum because python converts types correctly
        return sum([elem() for elem in self.children])

    @CompositeWrapper.from_graph_df
    def graph_df(self, wrt, grad):
        wrt_count = self.children.count(wrt)
        return grad * Variable(wrt_count)


def _initial_grad_wrapper(init_fn):
    def add_init_grad(self, node, wrt, initial_grad=None, name="", graph_expand=False):
        if initial_grad is None:
            initial_grad = Variable(1, name="init_grad")

        return init_fn(self, node, wrt, initial_grad=initial_grad, name=name, graph_expand=graph_expand)

    return add_init_grad


class Grad(CompositeOperation):
    @_initial_grad_wrapper
    def __init__(self, node, wrt, initial_grad, name="", graph_expand=False):
        super().__init__([node],
                         name=name + " grad w_val.r.t. '" + str(wrt) + "'",
                         graph_expand=graph_expand)
        self.wrt = wrt
        self.initial_grad = initial_grad
        self.out = self.init_graph()

    def graph(self):
        with self.children[0].mark_node_in_graph():
            reverse_sorted_nodes = self.children[0].reverse_topo_sort()

        dct = collections.defaultdict(list)
        dct[self.children[0]].append(self.initial_grad)

        for node in reverse_sorted_nodes:
            dct[node] = Add(*dct[node], name=node.name + "_grad_sum")
            if node == self.wrt:
                break

            for child in set(node.children):
                app = node.graph_df(wrt=child, grad=dct[node])
                dct[child].append(app)

        val = dct[self.wrt]
        if isinstance(val, Add):
            return val
        else:
            return Add(*val, name=self.wrt.name + "_grad_sum")


def checkpoint(fn):
    # how does wrapping grad_fn in a primitive look, since it depends on stuff other than whats passed
    # to it?
    def wrap_in_primitive(*fn_args, **kwargs):
        op = Primitive(children=fn_args, name=fn.__name__)

        op.eval = lambda: fn(*fn_args, **kwargs)()
        op.graph_df = lambda wrt, grad: grad_fn(fn(*fn_args, **kwargs), wrt, grad)

        return op

    return wrap_in_primitive


@checkpoint
def grad_fn(top_node, wrt, initial_grad=None):
    # it should be okay here since grad_fn is not an op? or is it?
    if initial_grad is None:
        initial_grad = Variable(1, name="init_grad")
    with top_node.mark_node_in_graph():
        reverse_sorted_nodes = top_node.reverse_topo_sort()

    dct = collections.defaultdict(list)
    dct[top_node].append(initial_grad)

    for node in reverse_sorted_nodes:
        dct[node] = Add(*dct[node], name=node.name + "_grad_sum")
        if node == wrt:
            break

        for child in set(node.children):
            app = node.graph_df(wrt=child, grad=dct[node])
            dct[child].append(app)

    val = dct[wrt]
    if isinstance(val, Add):
        return val
    else:
        return Add(*val, name=wrt.name + "_grad_sum")
