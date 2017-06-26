from unittest import TestCase

import tensorflow as tf

from core.ops import *


class TestOperation(TestCase):
    def setUp(self):
        np.random.seed(1337)
        self.x = np.random.rand(2, 3)
        self.w = np.random.rand(3, 5)

        self.tf_x = tf.constant(self.x)
        self.tf_w = tf.constant(self.w)
        self.tf_mul = self.tf_x @ self.tf_w
        self.tf_nonlin = tf.nn.sigmoid(self.tf_mul)

        self.var_x = Variable(name="x")
        self.var_w = Variable(name="w")
        self.var_mul = self.var_x @ self.var_w
        self.var_nonlin = Sigmoid(self.var_mul)

        self.input_dict = {"x": self.x, "w": self.w}

    def test_f(self):
        graph = self.var_nonlin
        with tf.Session():
            tf_val = self.tf_nonlin.eval()

        my_val = graph.f(self.input_dict)

        np.testing.assert_allclose(my_val, tf_val)

    def test_oneop_df_ww(self):
        graph = self.var_mul
        with tf.Session():
            tf_grads = tf.gradients(self.tf_mul, self.tf_w)[0].eval()

        graph.compute_derivatives(self.input_dict)
        my_grads = graph.accumulate_all_gradients(wrt="w")

        np.testing.assert_allclose(my_grads, tf_grads)

    def test_oneop_df_xx(self):
        graph = self.var_mul
        with tf.Session():
            tf_grads = tf.gradients(self.tf_mul, self.tf_x)[0].eval()

        graph.compute_derivatives(self.input_dict)
        my_grads = graph.accumulate_all_gradients(wrt="x")

        np.testing.assert_allclose(my_grads, tf_grads)

    def test_df_matmul(self):
        graph = self.var_nonlin
        with tf.Session():
            tf_grads = tf.gradients(self.tf_nonlin, self.tf_mul)[0].eval()

        graph.compute_derivatives(self.input_dict)
        my_grads = graph.accumulate_all_gradients(wrt="MatMul")

        np.testing.assert_allclose(my_grads, tf_grads)

    def test_df_ww(self):
        graph = self.var_nonlin
        with tf.Session():
            tf_grads = tf.gradients(self.tf_nonlin, self.tf_w)[0].eval()

        graph.compute_derivatives(self.input_dict)
        my_grads = graph.accumulate_all_gradients(wrt="w")

        np.testing.assert_allclose(my_grads, tf_grads)

    def test_df_xx(self):
        graph = self.var_nonlin
        with tf.Session():
            tf_grads = tf.gradients(self.tf_nonlin, self.tf_x)[0].eval()

        graph.compute_derivatives(self.input_dict)
        my_grads = graph.accumulate_all_gradients(wrt="x")

        np.testing.assert_allclose(my_grads, tf_grads)

    def test_df_with_scalar(self):
        graph = 5 * self.var_x
        tf_graph = 5 * self.tf_x
        with tf.Session():
            tf_grads = tf.gradients(tf_graph, self.tf_x)[0].eval()

        graph.compute_derivatives(self.input_dict)
        my_grads = graph.accumulate_all_gradients(wrt="x")

        np.testing.assert_allclose(my_grads, tf_grads)

    def test_df_transpose(self):
        graph = Transpose(self.var_x)
        tf_graph = tf.transpose(self.tf_x)
        with tf.Session():
            tf_grads = tf.gradients(tf_graph, self.tf_x)[0].eval()

        graph.compute_derivatives(self.input_dict)
        my_grads = graph.accumulate_all_gradients(wrt="x")

        np.testing.assert_allclose(my_grads, tf_grads)
