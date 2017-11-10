import numpy as np
import tensorflow as tf
import timeout_decorator
import automatic_differentiation as ad


def differentiate_n_times(my_graph, tf_graph, my_vars, tf_vars, order=1, my_curr_grad=None, tf_curr_grad=None):
    # all vars share the first graph derivation and speed up the test
    # but higher order graphs are different

    for i in range(order):
        if i == 0:
            my_graphs = ad.grad(my_graph, my_vars, previous_grad=my_curr_grad)
            if tf_graph is not None:
                tf_graphs = tf.gradients(tf_graph, tf_vars, grad_ys=tf_curr_grad)
        else:
            for i in range(len(my_graphs)):
                my_graphs[i] = ad.grad(my_graphs[i], [my_vars[i]])[0]
                if tf_graphs[i] is not None:
                    tf_graphs[i] = tf.gradients(tf_graphs[i], tf_vars[i])[0]
    return my_graphs, tf_graphs


def eval_graphs(my_graph, tf_graph, my_var, tf_var, n):
    tf_grads = 0
    if tf_graph is not None:
        tf_grads = tf_graph.eval()
    my_grads = my_graph()

    print("---------- " + str(n) + "df w.r.t. " + str(my_var) + " ----------")
    print("My_val:", my_grads)
    print("Tf_val:", tf_grads)
    my_val = np.broadcast_to(my_grads, my_var.shape)
    tf_val = np.broadcast_to(tf_grads, my_var.shape)
    np.testing.assert_allclose(my_val, tf_val, rtol=1e-2, atol=1e-6)


def oneop_df_n_times(test, my_graph, tf_graph, my_vars, tf_vars, n=1, my_curr_grad=None, tf_curr_grad=None):
    my_graphs, tf_graphs = differentiate_n_times(my_graph, tf_graph, my_vars, tf_vars, order=n,
                                                 my_curr_grad=my_curr_grad, tf_curr_grad=tf_curr_grad)

    for my_var, tf_var, my_graph, tf_graph in zip(my_vars, tf_vars, my_graphs, tf_graphs):
        name = str(n) + "df_wrt_" + my_var.name
        if my_graph.name == "extra_exp_op":
            name += " with input gradient!!"
        with test.subTest(name):
            eval_graphs(my_graph, tf_graph, my_var, tf_var, n)


def oneop_f(my_graph, tf_graph):
    tf_val = tf_graph.eval()
    my_val = my_graph()

    print("---------- f ----------")
    print("My_val:", my_val)
    print("Tf_val:", tf_val)
    np.testing.assert_allclose(my_val, tf_val)


def test_one_op(test, my_graph, tf_graph, my_wrt, tf_wrt, order=1):
    """
    Evaluates f and arbitrarily higher order df of my graph and tf graph (all in different subtests)
    and compares the results.
    """
    with tf.Session():
        with test.subTest("f"):
            oneop_f(my_graph, tf_graph)

        for deriv in range(1, order + 1):
            oneop_df_n_times(test, my_graph, tf_graph, my_wrt, tf_wrt, n=deriv)

        print("#############################")
        print("    with random curr_grad    ")
        print("#############################")

        my_extended_graph = ad.Exp(my_graph, name="extra_exp_op")
        tf_extended_graph = tf.exp(tf_graph, name="extra_exp_op")
        for deriv in range(1, order + 1):
            oneop_df_n_times(test, my_extended_graph, tf_extended_graph, my_wrt, tf_wrt, n=deriv)


def custom_test(test, my_graph, my_wrt_vars, tf_graph=None, tf_wrt_vars=None):
    test_one_op(test, my_graph, tf_graph, my_wrt_vars, tf_wrt_vars)
    # ad.test_numeric(test, my_graph, my_wrt_vars)
