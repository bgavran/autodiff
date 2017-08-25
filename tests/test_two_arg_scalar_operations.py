from unittest import TestCase

import tensorflow as tf

from core.ops import *


class TestTwoArgScalarOperations(TestCase):
    def setUp(self):
        np.random.seed(1337)
        self.w0 = np.random.rand()
        self.w1 = np.random.rand()

        self.tf_w0 = tf.constant(self.w0)
        self.tf_w1 = tf.constant(self.w1)

        self.var_w0 = Variable(name="w0")
        self.var_w1 = Variable(name="w1")

        self.input_dict = {"w0": self.w0, "w1": self.w1}

    def oneop_f(self, var_op, tf_op):
        tf_graph = tf_op(self.tf_w0, self.tf_w1)
        graph = var_op(self.var_w0, self.var_w1)

        with tf.Session():
            tf_val = tf_graph.eval()

        my_val = graph.eval(self.input_dict)

        np.testing.assert_allclose(my_val, tf_val)

    def oneop_df_wrt_w0(self, var_op, tf_op):
        tf_graph = tf_op(self.tf_w0, self.tf_w1)
        graph = var_op(self.var_w0, self.var_w1)
        with tf.Session():
            tf_grads = tf.gradients(tf_graph, self.tf_w0)[0].eval()

        grad_ops = Grad(graph, wrt="w0")
        my_grads = grad_ops.eval(self.input_dict)

        np.testing.assert_allclose(my_grads, tf_grads)

    def oneop_df_wrt_w1(self, var_op, tf_op):
        tf_graph = tf_op(self.tf_w0, self.tf_w1)
        graph = var_op(self.var_w0, self.var_w1)
        with tf.Session():
            tf_grads = tf.gradients(tf_graph, self.tf_w1)[0].eval()

        grad_ops = Grad(graph, wrt="w1")
        my_grads = grad_ops.eval(self.input_dict)

        np.testing.assert_allclose(my_grads, tf_grads)

    def oneop(self, var_op, tf_op):
        with self.subTest("f"):
            self.oneop_f(var_op, tf_op)
        with self.subTest("df_wrt_w0"):
            self.oneop_df_wrt_w0(var_op, tf_op)
        with self.subTest("df_wrt_w1"):
            self.oneop_df_wrt_w1(var_op, tf_op)

    def test_add(self):
        self.oneop(Add, tf.add)

    def test_mul(self):
        self.oneop(Mul, tf.multiply)

    def test_power(self):
        self.oneop(Pow, tf.pow)

    def test_squared_difference(self):
        self.oneop(SquaredDifference, tf.squared_difference)
