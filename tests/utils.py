import numpy as np
import tensorflow as tf
import timeout_decorator
import automatic_differentiation as ad


def differentiate_n_times(my_graph, tf_graph, my_vars, tf_vars, n=1):
    # all vars share the first graph derivation and speed up the test
    # higher order graphs are different

    for i in range(n):
        if i == 0:
            my_graphs = ad.grad(my_graph, my_vars)
            if tf_graph is not None:
                tf_graphs = tf.gradients(tf_graph, tf_vars)
        else:
            for i in range(len(my_graphs)):
                my_graphs[i] = ad.grad(my_graphs[i], [my_vars[i]])[0]
                if tf_graphs[i] is not None:
                    tf_graphs[i] = tf.gradients(tf_graphs[i], tf_vars[i])[0]
    return my_graphs, tf_graphs


def oneop_f(my_graph, tf_graph):
    tf_val = tf_graph.eval()
    my_val = my_graph()

    print("---------- f ----------")
    print("My_val:", my_val)
    print("Tf_val:", tf_val)
    print("-----------------------")
    np.testing.assert_allclose(my_val, tf_val)


def oneop_df_n_times(test, my_graph, tf_graph, my_vars, tf_vars, n=1):
    my_graphs, tf_graphs = differentiate_n_times(my_graph, tf_graph, my_vars, tf_vars, n=n)

    for my_var, tf_var, my_graph, tf_graph in zip(my_vars, tf_vars, my_graphs, tf_graphs):
        with test.subTest(str(n) + "df_wrt_" + my_var.name):
            eval_graphs(my_graph, tf_graph, my_var, tf_var, n)


def eval_graphs(my_graph, tf_graph, my_var, tf_var, n):
    tf_grads = 0
    if tf_graph is not None:
        tf_grads = tf_graph.eval()
    my_grads = my_graph()

    print("---------- " + str(n) + "df w.r.t. " + str(my_var) + " ----------")
    print("My_val:", my_grads)
    print("Tf_val:", tf_grads)
    my_val = my_grads + my_var()
    tf_val = tf_grads + tf_var.eval()
    np.testing.assert_allclose(my_val, tf_val)


@timeout_decorator.timeout(1)
def test_one_op(test, my_graph, tf_graph, my_wrt, tf_wrt, n=2):
    """
    Evaluates f and arbitrarily higher order df of my graph and tf graph (all in different subtests)
    and compares the results.
    """
    with tf.Session():
        with test.subTest("f"):
            oneop_f(my_graph, tf_graph)

        for deriv in range(1, n + 1):
            oneop_df_n_times(test, my_graph, tf_graph, my_wrt, tf_wrt, n=deriv)
