import numpy as np
import collections
from automatic_differentiation.src.core.utils import reverse_topo_sort
from automatic_differentiation.src.core.ops import Add
from automatic_differentiation.src.core.computational_graph import Variable


def add_sum_name(node):
    return "'" + node.name + "' grad_sum"


def grad(top_node, wrt_list, previous_grad=None):
    assert isinstance(wrt_list, list) or isinstance(wrt_list, tuple)
    if previous_grad is None:
        previous_grad = Variable(np.ones(top_node.shape), name=add_sum_name(top_node))

    dct = collections.defaultdict(list)
    dct[top_node].append(previous_grad)

    for node in reverse_topo_sort(top_node):
        dct[node] = Add(*dct[node], name=add_sum_name(node))

        for child in set(node.children):
            app = node.partial_derivative(wrt=child, previous_grad=dct[node])
            dct[child].append(app)

    return [dct[wrt] if isinstance(dct[wrt], Add) else Add(*dct[wrt], name=add_sum_name(wrt)) for wrt in wrt_list]
