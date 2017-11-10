import numpy as np
import timeout_decorator
import automatic_differentiation as ad


def uncache_subtree(node):
    node.cached = None
    for child in node.children:
        uncache_subtree(child)


def numerical_gradient(graph, wrt_var, eps=1e-6):
    old_val = wrt_var().copy()
    grad = np.zeros_like(old_val)
    it = np.nditer(old_val, flags=['multi_index'])
    while not it.finished:
        ind = it.multi_index
        med = old_val[ind]

        uncache_subtree(graph)
        wrt_var.cached = old_val
        wrt_var.cached[ind] = med + eps
        graph_val_higher = graph().copy()

        uncache_subtree(graph)
        wrt_var.cached = old_val
        wrt_var.cached[ind] = med - eps
        graph_val_lower = graph().copy()

        grad[ind] = np.sum((graph_val_higher - graph_val_lower) / (2 * eps))
        it.iternext()

    uncache_subtree(graph)
    return grad


def differentiate_n_times_num(graph, wrt_vars, order):
    if order == 1:
        backprop_graphs = ad.grad(graph, wrt_vars)
        numeric_grads = [numerical_gradient(graph, var) for var in wrt_vars]
    else:
        for o in range(order):
            if o == 0:
                backprop_graphs = ad.grad(graph, wrt_vars)
            else:
                if o == order - 1:
                    numeric_grads = [numerical_gradient(graph, var) for graph, var in zip(backprop_graphs, wrt_vars)]
                backprop_graphs = [ad.grad(graph, [var])[0] for graph, var in zip(backprop_graphs, wrt_vars)]
    return backprop_graphs, numeric_grads


def numerical_check(test, graph, wrt_vars, order=1):
    backprop_graphs, numeric_grads = differentiate_n_times_num(graph, wrt_vars, order=order)

    for wrt_var, graph_grad, num_grad in zip(wrt_vars, backprop_graphs, numeric_grads):
        name = "num" + str(order) + "df_wrt_" + wrt_var.name
        if graph.name == "extra_exp_op":
            name += " as input to another op!!!"
        with test.subTest(name):
            print("---------- " + name + " ----------")
            print("Backprop grad:", graph_grad())
            print("Numeric grad:", num_grad)
            broadcasted_grad = np.broadcast_to(graph_grad(), wrt_var().shape)  # not necessarily the same shape
            np.testing.assert_allclose(broadcasted_grad, num_grad, rtol=1e-2, atol=1e-6)


def test_numeric(test, graph, wrt_vars, order=1):
    for deriv in range(1, order + 1):
        numerical_check(test, graph, wrt_vars, order=deriv)

    extended_graph = ad.Exp(graph, name="extra_exp_op")
    for deriv in range(1, order + 1):
        numerical_check(test, extended_graph, wrt_vars, order=deriv)
