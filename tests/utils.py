import numpy as np
import tensorflow as tf
import timeout_decorator
import automatic_differentiation as ad


def differentiate_n_times(my_graph, tf_graph, my_var, tf_var, n=1):
    for _ in range(n):
        my_graph = ad.Grad(my_graph, wrt=my_var)
        if tf_graph is not None:
            tf_graph = tf.gradients(tf_graph, tf_var)[0]
    return my_graph, tf_graph


def oneop_f(my_graph, tf_graph):
    tf_val = tf_graph.eval()
    my_val = my_graph.eval()

    print("---------- f ----------")
    print("My_val:", my_val)
    print("Tf_val:", tf_val)
    print("-----------------------")
    np.testing.assert_allclose(my_val, tf_val)


def oneop_df_n_times(test, my_graph, tf_graph, my_var, tf_var, n=1):
    my_graph, tf_graph = differentiate_n_times(my_graph, tf_graph, my_var, tf_var, n=n)

    tf_grads = 0
    if tf_graph is not None:
        tf_grads = tf_graph.eval()
    my_grads = my_graph.eval()

    print("---------- " + str(n) + "df ----------")
    print("My_val:", my_grads)
    print("Tf_val:", tf_grads)
    my_val = my_grads + my_var()
    tf_val = tf_grads + tf_var.eval()
    np.testing.assert_allclose(my_val, tf_val)


@timeout_decorator.timeout(1)
def test_one_op(test, my_graph, tf_graph, my_wrt, tf_wrt, n=3):
    """
    Evaluates f and arbitrarily higher order df of my graph and tf graph (all in different subtests)
    and compares the results.
    """
    with tf.Session():
        with test.subTest("f"):
            oneop_f(my_graph, tf_graph)

        for deriv in range(1, n + 1):
            for i, wrt in enumerate(my_wrt):
                with test.subTest(str(deriv) + "df_wrt_" + wrt.name):
                    oneop_df_n_times(test, my_graph, tf_graph, my_wrt[i], tf_wrt[i], n=deriv)
