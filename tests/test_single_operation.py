from unittest import TestCase

import tensorflow as tf

from core.ops import *


class TestOneArgOperations(TestCase):
    def setUp(self):
        np.random.seed(1337)
        self.w0 = np.random.rand(2, 3)

        self.tf_w0 = tf.constant(self.w0)
        self.var_w0 = Variable(name="w0")

        self.input_dict = {"w0": self.w0}

    def oneop_f(self, var_op, tf_op):
        tf_graph = tf_op(self.tf_w0)
        graph = var_op(self.var_w0)

        with tf.Session():
            tf_val = tf_graph.eval()

        my_val = graph.f(self.input_dict)

        np.testing.assert_allclose(my_val, tf_val)

    def oneop_df_wrt_w0(self, var_op, tf_op):
        tf_graph = tf_op(self.tf_w0)
        graph = var_op(self.var_w0)
        with tf.Session():
            tf_grads = tf.gradients(tf_graph, self.tf_w0)[0].eval()

        graph.compute_derivatives(self.input_dict)
        my_grads = graph.accumulate_all_gradients(wrt="w0")

        np.testing.assert_allclose(my_grads, tf_grads)

    def oneop(self, var_op, tf_op):
        with self.subTest("f"):
            self.oneop_f(var_op, tf_op)
        with self.subTest("df"):
            self.oneop_df_wrt_w0(var_op, tf_op)

    def test_sigmoid(self):
        self.oneop(Sigmoid, tf.nn.sigmoid)

    def test_relu(self):
        self.oneop(ReLU, tf.nn.relu)

    # def test_gauss(self):
    #     # need to fix dtypes?
    #     self.oneop(Gauss, tf.contrib.distributions.Normal(loc=0., scale=1.).prob)

    def test_transpose(self):
        self.oneop(Transpose, tf.transpose)

    def test_recipr(self):
        self.oneop(Recipr, tf.reciprocal)

    def test_negate(self):
        self.oneop(Negate, tf.negative)


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

        my_val = graph.f(self.input_dict)

        np.testing.assert_allclose(my_val, tf_val)

    def oneop_df_wrt_w0(self, var_op, tf_op):
        tf_graph = tf_op(self.tf_w0, self.tf_w1)
        graph = var_op(self.var_w0, self.var_w1)
        with tf.Session():
            tf_grads = tf.gradients(tf_graph, self.tf_w0)[0].eval()

        graph.compute_derivatives(self.input_dict)
        my_grads = graph.accumulate_all_gradients(wrt="w0")

        np.testing.assert_allclose(my_grads, tf_grads)

    def oneop_df_wrt_w1(self, var_op, tf_op):
        tf_graph = tf_op(self.tf_w0, self.tf_w1)
        graph = var_op(self.var_w0, self.var_w1)
        with tf.Session():
            tf_grads = tf.gradients(tf_graph, self.tf_w1)[0].eval()

        graph.compute_derivatives(self.input_dict)
        my_grads = graph.accumulate_all_gradients(wrt="w1")

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

    def test_squared_difference(self):
        self.oneop(SquaredDifference, tf.squared_difference)
