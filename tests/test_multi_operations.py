from unittest import TestCase

import tensorflow as tf

from core.ops import *

from tests import utils


class TestOperation(TestCase):
    def setUp(self):
        """
        Graph looks like this:
        x_val    w_val
         \  /
         MatMul
          |
        Sigmoid

        x_val.shape = (2, 3)
        w_val.shape = (3, 5)
        MatMul.shape = (2, 5)
        Sigmoid.shape = (2, 5)

        """
        np.random.seed(1337)
        self.x_val = np.random.randn(2, 3)
        self.w_val = np.random.randn(3, 5)

        self.tf_x = tf.placeholder(dtype=tf.float64)
        self.tf_w = tf.placeholder(dtype=tf.float64)
        self.tf_mul = self.tf_x @ self.tf_w
        self.tf_nonlin = tf.nn.sigmoid(self.tf_mul)

        self.my_x = Variable(self.x_val, name="x_val")
        self.my_w = Variable(self.w_val, name="w_val")
        self.var_mul = self.my_x @ self.my_w
        self.var_nonlin = Sigmoid(self.var_mul)

        self.tf_input_dict = {self.tf_x: self.x_val, self.tf_w: self.w_val}

    def test_f(self):
        graph = self.var_nonlin
        with tf.Session():
            tf_val = self.tf_nonlin.eval(self.tf_input_dict)

        my_val = graph.eval()

        np.testing.assert_allclose(my_val, tf_val)

    def test_oneop_df_w(self):
        graph = self.var_mul
        with tf.Session():
            tf_grads = tf.gradients(self.tf_mul, self.tf_w)[0].eval(self.tf_input_dict)

        grad_ops = Grad(graph, self.my_w)
        my_grads = grad_ops.eval()

        np.testing.assert_allclose(my_grads + self.w_val, tf_grads + self.w_val)

    def test_oneop_df_x(self):
        graph = self.var_mul
        with tf.Session():
            tf_grads = tf.gradients(self.tf_mul, self.tf_x)[0].eval(self.tf_input_dict)

        grad_ops = Grad(graph, self.my_x)
        my_grads = grad_ops.eval()

        # This is still kind of a dirty hack?
        np.testing.assert_allclose(my_grads + self.x_val, tf_grads + self.x_val)

    def test_df_matmul(self):
        graph = self.var_nonlin
        with tf.Session():
            tf_grads = tf.gradients(self.tf_nonlin, self.tf_mul)[0].eval(self.tf_input_dict)

        grad_ops = Grad(graph, wrt=self.var_mul)
        my_grads = grad_ops.eval()

        np.testing.assert_allclose(my_grads, tf_grads)

    def test_df_w(self):
        graph = self.var_nonlin
        with tf.Session():
            tf_grads = tf.gradients(self.tf_nonlin, self.tf_w)[0].eval(self.tf_input_dict)

        grad_ops = Grad(graph, self.my_w)
        my_grads = grad_ops.eval()

        np.testing.assert_allclose(my_grads, tf_grads)

    def test_df_x(self):
        graph = self.var_nonlin

        my_graph, tf_graph = utils.differentiate_n_times(graph, self.tf_nonlin, self.my_x, self.tf_x, n=1)
        with tf.Session():
            if tf_graph is not None:
                tf_grads = tf_graph.eval(feed_dict=self.tf_input_dict)
            else:
                tf_grads = 0
        my_grads = my_graph.eval()

        np.testing.assert_allclose(my_grads, tf_grads)

    def test_df_with_scalar(self):
        graph = 5 * self.my_x
        tf_graph = 5 * self.tf_x
        with tf.Session():
            tf_grads = tf.gradients(tf_graph, self.tf_x)[0].eval(self.tf_input_dict)

        grad_ops = Grad(graph, self.my_x)
        my_grads = grad_ops.eval()

        np.testing.assert_allclose(my_grads, tf_grads)

    def test_df_transpose(self):
        graph = Transpose(self.my_x)
        tf_graph = tf.transpose(self.tf_x)
        with tf.Session():
            tf_grads = tf.gradients(tf_graph, self.tf_x)[0].eval(self.tf_input_dict)

        grad_ops = Grad(graph, wrt=self.my_x)
        my_grads = grad_ops.eval()

        # TODO this might not be such a good idea?
        # Is there a way to dynamically get the shape of the gradient?
        # Here I'm just relying on broadcasting to work
        # But it does seem like the natural way?
        np.testing.assert_allclose(my_grads + self.x_val, tf_grads + self.x_val)
