import numpy as np
import tensorflow as tf
from core.computational_graph import Grad


def differentiate_n_times(my_graph, tf_graph, my_var, tf_var, n=1):
    for _ in range(n):
        my_graph = Grad(my_graph, wrt=my_var)
        if tf_graph is not None:
            tf_graph = tf.gradients(tf_graph, tf_var)[0]
    return my_graph, tf_graph


def oneop_f(my_graph, tf_graph):
    with tf.Session():
        tf_val = tf_graph.eval()

    my_val = my_graph.eval()

    print("---------- f ----------")
    print("My_val:", my_val)
    print("Tf_val:", tf_val)
    print("-----------------------")
    np.testing.assert_allclose(my_val, tf_val)


def oneop_df_n_times(test, my_graph, tf_graph, wrt_vars, n=1):
    my_var, tf_var = wrt_vars

    my_graph, tf_graph = differentiate_n_times(my_graph, tf_graph, my_var, tf_var, n=n)

    with tf.Session():
        if tf_graph is not None:
            tf_grads = tf_graph.eval()
        else:
            tf_grads = 0
    my_grads = my_graph.eval()

    print("---------- " + str(n) + "df ----------")
    print("My_val:", my_grads)
    print("Tf_val:", tf_grads)
    my_val = my_grads + my_var()
    with tf.Session():
        tf_val = tf_grads + tf_var.eval()
    np.testing.assert_allclose(my_val, tf_val)
