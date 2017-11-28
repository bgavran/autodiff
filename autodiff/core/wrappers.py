from .node import Node
from .grad import grad


def checkpoint(fn):
    def wrap_in_primitive(*fn_args):
        op = Node(children=fn_args, name=fn.__name__)

        op._eval = lambda: fn(*fn_args)()
        op._partial_derivative = lambda wrt, previous_grad: grad(fn(*fn_args), [wrt], previous_grad=previous_grad)[0]
        # should graph_df return the already called Node() or just the Node, like right now?

        return op

    return wrap_in_primitive


