import automatic_differentiation as ad


def checkpoint(fn):
    def wrap_in_primitive(*fn_args):
        op = ad.Primitive(children=fn_args, name=fn.__name__)

        op.f = lambda: fn(*fn_args)()
        op.graph_df = lambda wrt, curr_grad: ad.grad(fn(*fn_args), [wrt], curr_grad=curr_grad)[0]
        # should graph_df return the already called Node() or just the Node, like right now?

        return op

    return wrap_in_primitive
