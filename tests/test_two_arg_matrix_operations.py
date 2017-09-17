from unittest import TestCase

import tensorflow as tf

from core.ops import *
from tests import utils


class TestTwoArgScalarOperations(TestCase):
    def setUp(self):
        np.random.seed(1337)

        self.w0 = np.random.randn(2, 3)
        self.w1 = np.random.randn(3, 5)

        self.tf_w0 = tf.placeholder(dtype=tf.float64)
        self.tf_w1 = tf.placeholder(dtype=tf.float64)

        self.var_w0 = Variable(name="w0")
        self.var_w1 = Variable(name="w1")

        self.my_input_dict = {self.var_w0: self.w0, self.var_w1: self.w1}
        self.tf_input_dict = {self.tf_w0: self.w0, self.tf_w1: self.w1}

    def oneop_df_n_times(self, var_op, tf_op, wrt_vars, n=1):
        my_var, tf_var = wrt_vars

        tf_graph = tf_op(self.tf_w0, self.tf_w1)
        my_graph = var_op(self.var_w0, self.var_w1)

        my_graph, tf_graph = utils.differentiate_n_times(my_graph, tf_graph, my_var, tf_var, n=n)

        with tf.Session():
            if tf_graph is not None:
                tf_grads = tf_graph.eval(feed_dict=self.tf_input_dict)
            else:
                tf_grads = 0
        my_grads = my_graph.eval(self.my_input_dict)

        print("---------- " + str(n) + "df ----------")
        print("My_val:", my_grads)
        print("Tf_val:", tf_grads)
        my_val = my_grads + self.my_input_dict[my_var]
        tf_val = tf_grads + self.tf_input_dict[tf_var]
        np.testing.assert_allclose(my_val, tf_val)

    def oneop_f(self, var_op, tf_op):
        tf_graph = tf_op(self.tf_w0, self.tf_w1)
        graph = var_op(self.var_w0, self.var_w1)

        with tf.Session():
            tf_val = tf_graph.eval(self.tf_input_dict)

        my_val = graph.eval(self.my_input_dict)

        print("---------- input ----------")
        print("w0:", self.w0)
        print("---------------------------")

        print("---------- f ----------")
        print("My_val:", my_val)
        print("Tf_val:", tf_val)
        print("-----------------------")
        np.testing.assert_allclose(my_val, tf_val)

    def oneop(self, var_op, tf_op):
        print("---------- " + "inputs" + "   ----------")
        print("w0:", self.w0)
        print("w1:", self.w1)
        print("-------------------------")
        with self.subTest("f"):
            self.oneop_f(var_op, tf_op)
        with self.subTest("df_wrt_w0"):
            self.oneop_df_n_times(var_op, tf_op, [self.var_w0, self.tf_w0], n=1)
        with self.subTest("df_wrt_w1"):
            self.oneop_df_n_times(var_op, tf_op, [self.var_w1, self.tf_w1], n=1)

        with self.subTest("2df_wrt_w0"):
            self.oneop_df_n_times(var_op, tf_op, [self.var_w0, self.tf_w0], n=2)
        with self.subTest("2df_wrt_w1"):
            self.oneop_df_n_times(var_op, tf_op, [self.var_w1, self.tf_w1], n=2)

        with self.subTest("3df_wrt_w0"):
            self.oneop_df_n_times(var_op, tf_op, [self.var_w0, self.tf_w0], n=3)
        with self.subTest("3df_wrt_w1"):
            self.oneop_df_n_times(var_op, tf_op, [self.var_w1, self.tf_w1], n=3)

    def test_matmul(self):
        self.oneop(MatMul, tf.matmul)
