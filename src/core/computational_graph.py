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
        return "//".join(self.context + [self.name])

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


class Operation(Node):
    epsilon = 1e-12

    @Node._constant_wrapper
    def __init__(self, children, name=""):
        super().__init__(name)
        self.children = children

        for child in set(self.children):
            child.parents.add(self)
        self.parents = set()
        self.visited_topo_sort = False
        self.temporary_mark = False

    def apply_on_self_and_children(self, fn, *args, **kwargs):
        fn(self, *args, **kwargs)
        for child in set(self.children):
            Operation.apply_on_self_and_children(child, fn, *args, **kwargs)

    """
    Are there any other problems? Is there a problem with node's parents? 
    Perhaps interfaces are still needed?
    How many parents do CompositeOperation's children have? Just the CompositeOperation?
    Then the bottom node in CompOp has a child which is the child of CompOp but that child doesn't have the
    parent which is the bottom node, but the CompOp node? That seems to make sense.
    That means the Operation just adds parents if the child node has the same context.
    
    But the nodes added in Grad.graph() operation will have children nodes INSIDE the graph()! Do we want that?
    
    ---
    Update: so it seems to work partially, the problem is we still need JUST the real children.
    Passing all nodes to grad makes the grad be calculated w.r.t. children we don't really need it 
    calculated.
    But to know the real children, we first need to calculate the gradient's graph()...
    Also by passing all children to grad, they have wrong parents also (in gradient's graph()) ?
    ---
    So what is the solution?
    CompOp CAN depend on stuff other than x, we recompute the real children after (BFS).
    There's no interfaces, but node's parents are needed to efficiently compute topological sort.
    Sometimes there's a problem with abstarction with Grad (we don't want to display stuff inside a CompOp
    whose Grad we're taking if its not set to be expanded while graphed).
    
    ---
    Other problem:
    Too slow?
    
    
    
    """

    def __iter__(self):
        yield self
        for child in set(self.children):
            yield from child

    def find_nodes(self, with_context=True):
        """
        Traces back throughout the nodes children and returns of a list of nodes.
        The elements of the list depend on with_context variable.
        If with_context=True:
            returns nodes with the same context
        else:
            returns just the top nodes that have different context (not their children)

        """
        ll = []
        if with_context:
            ll.append(self)

        for child in self.children:
            if child.context == self.context:
                ll.extend(child.find_nodes(with_context))
            else:
                if not with_context:
                    ll.append(child)
        return set(ll)

    def reverse_topo_sort(self):
        nodes_in_graph = list(self)  # list of nodes this node depends on (we ignore other parents)
        ll = []

        def visit(node):
            if node.visited_topo_sort:
                return
            assert node.temporary_mark is False  # must not be a cyclic graph
            node.temporary_mark = True
            for parent in node.parents:
                if parent in nodes_in_graph:  # slow?
                    visit(parent)

            node.visited_topo_sort = True
            ll.append(node)

        for node in nodes_in_graph:
            if not node.visited_topo_sort:
                visit(node)

        def fn(instance):
            instance.visited_topo_sort = False
            instance.temporary_mark = False

        self.apply_on_self_and_children(fn)

        return ll

    def all_nodes(self):
        return set(self)

    def find_node_by_id(self, node_id):
        for node in self.all_nodes():
            if node.id == node_id:
                return node
        raise ValueError("Node with id", node_id, "doesn't exist!")

    def eval(self, input_dict):
        raise NotImplementedError()

    def graph_df(self, wrt, grad):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.eval(*args, **kwargs)

    def add_node_subgraph_to_plot_graph(self, digraph):
        if str(self.id) not in digraph.set_of_added_nodes:
            # Add node
            digraph.add_node_with_context(self, self.context)

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

    def graph_df(self, wrt, grad):
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

    def graph_df(self, wrt, grad):
        if self == wrt:
            return grad
        return 0


class CompositeWrapper:
    @staticmethod
    def from_graph_df(fn):
        def wrap_in_composite(instance, wrt, grad, expand_when_graphed=True):
            name = "Gradient graph of " + instance.name + " "
            children = [grad] + list(set(instance.children))

            op = CompositeOperation(children,
                                    name=name,
                                    expand_when_graphed=expand_when_graphed)

            op.graph = lambda: fn(instance, wrt, grad)
            op.out = op.init_graph()

            return op

        return wrap_in_composite

    @staticmethod
    def from_function(fn):
        def wrap_in_composite(*fn_args, expand_when_graphed=False):
            op = CompositeOperation(interface=fn_args,
                                    name=fn.__name__,
                                    expand_when_graphed=expand_when_graphed)

            op.graph = lambda: fn(*op.children)
            op.out = op.init_graph()

            return op

        return wrap_in_composite


class CompositeOperation(Operation):
    def __init__(self, interface, name="", expand_when_graphed=True):
        """

        :param interface:
        :param name: 
        """
        super().__init__(interface, name)
        self.expand_when_graphed = expand_when_graphed
        self._out = None

    def init_graph(self):
        ctx = self.name + " " + str(self.id)
        with Node.add_context(ctx):
            out = self.graph()
        assert out is not None

        # this is needed only for Grad. Can it be a bit cleaner?
        self.children = out.find_nodes(with_context=False)
        for child in set(self.children):
            child.parents.add(self)

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
        if self.expand_when_graphed:
            return self.out.get_node_for_graph()
        else:
            return self

    def add_node_subgraph_to_plot_graph(self, digraph):
        if self.expand_when_graphed:
            self.out.add_node_subgraph_to_plot_graph(digraph)
        else:
            Operation.add_node_subgraph_to_plot_graph(self, digraph)

    def eval(self, input_dict):
        return self.out.eval(input_dict)

    # @CompositeWrapper.from_graph_df
    def graph_df(self, wrt, grad):
        return Grad(self.out, wrt=wrt, initial_grad=grad, name=self.name)

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
    def graph_df(self, wrt, grad):
        wrt_count = self.children.count(wrt)
        return grad * Constant(wrt_count)


class Grad(CompositeOperation):
    def __init__(self, node, wrt, initial_grad=None, name="", expand_when_graphed=True):
        super().__init__([node],
                         name=name + " grad w.r.t. " + str(wrt),
                         expand_when_graphed=expand_when_graphed)
        self.wrt = wrt
        self.initial_grad = Constant(1, name=node.name + "_grad") if initial_grad is None else initial_grad
        self.out = self.init_graph()

    def graph(self):
        reverse_sorted_nodes = self.children[0].reverse_topo_sort()

        if self.wrt not in reverse_sorted_nodes:
            return Constant(0, name=str(self.wrt) + "_zero")

        dct = collections.defaultdict(list)
        dct[reverse_sorted_nodes[0].id].append(self.initial_grad)

        for node in reverse_sorted_nodes:
            dct[node.id] = Add(*dct[node.id], name=node.name + "_grad_sum")

            for child in set(node.children):
                app = node.graph_df(child, grad=dct[node.id])
                dct[child.id].append(app)

        return dct[self.wrt.id]
