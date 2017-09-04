import tensorflow as tf
from core.computational_graph import Grad


def differentiate_n_times(my_graph, tf_graph, my_var, tf_var, n=1):
    for _ in range(n):
        my_graph = Grad(my_graph, wrt=my_var)
        if tf_graph is not None:
            tf_graph = tf.gradients(tf_graph, tf_var)[0]
    return my_graph, tf_graph
