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

        my_val = graph.eval(self.input_dict)

        np.testing.assert_allclose(my_val, tf_val)

    def test_oneop_df_w(self):
        graph = self.var_mul
        with tf.Session():
            tf_grads = tf.gradients(self.tf_mul, self.tf_w)[0].eval()

        grad_ops = Grad(graph, wrt="w")
        my_grads = grad_ops.eval(self.input_dict)

        np.testing.assert_allclose(my_grads + self.w, tf_grads + self.w)

    def test_oneop_df_x(self):
        graph = self.var_mul
        with tf.Session():
            tf_grads = tf.gradients(self.tf_mul, self.tf_x)[0].eval()

        grad_ops = Grad(graph, wrt="x")
        my_grads = grad_ops.eval(self.input_dict)

        # This is still kind of a dirty hack?
        np.testing.assert_allclose(my_grads + self.x, tf_grads + self.x)

    def test_df_matmul(self):
        graph = self.var_nonlin
        with tf.Session():
            tf_grads = tf.gradients(self.tf_nonlin, self.tf_mul)[0].eval()

        grad_ops = Grad(graph, wrt="MatMul")
        my_grads = grad_ops.eval(self.input_dict)

        np.testing.assert_allclose(my_grads, tf_grads)

    def test_df_w(self):
        graph = self.var_nonlin
        with tf.Session():
            tf_grads = tf.gradients(self.tf_nonlin, self.tf_w)[0].eval()

        grad_ops = Grad(graph, wrt="w")
        my_grads = grad_ops.eval(self.input_dict)

        np.testing.assert_allclose(my_grads, tf_grads)

    def test_df_x(self):
        graph = self.var_nonlin
        with tf.Session():
            tf_grads = tf.gradients(self.tf_nonlin, self.tf_x)[0].eval()

        grad_ops = Grad(graph, wrt="x")
        my_grads = grad_ops.eval(self.input_dict)

        np.testing.assert_allclose(my_grads, tf_grads)

    def test_df_with_scalar(self):
        graph = 5 * self.var_x
        tf_graph = 5 * self.tf_x
        with tf.Session():
            tf_grads = tf.gradients(tf_graph, self.tf_x)[0].eval()

        grad_ops = Grad(graph, wrt="x")
        my_grads = grad_ops.eval(self.input_dict)

        np.testing.assert_allclose(my_grads, tf_grads)

    def test_df_transpose(self):
        graph = Transpose(self.var_x)
        tf_graph = tf.transpose(self.tf_x)
        with tf.Session():
            tf_grads = tf.gradients(tf_graph, self.tf_x)[0].eval()

        grad_ops = Grad(graph, wrt="x")
        my_grads = grad_ops.eval(self.input_dict)

        # TODO this might not be such a good idea?
        # Is there a way to dynamically get the shape of the gradient?
        # Here I'm just relying on broadcasting to work
        np.testing.assert_allclose(my_grads + self.x, tf_grads + self.x)
