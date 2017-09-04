from unittest import TestCase

import tensorflow as tf

from core.ops import *

from tests import utils


class TestOperation(TestCase):
    def setUp(self):
        """
        Graph looks like this:
        x    w
         \  /
         MatMul
          |
        Sigmoid

        x.shape = (2, 3)
        w.shape = (3, 5)
        MatMul.shape = (2, 5)
        Sigmoid.shape = (2, 5)

        """
        np.random.seed(1337)
        self.x = np.random.rand(2, 3)
        self.w = np.random.rand(3, 5)

        self.tf_x = tf.placeholder(dtype=tf.float64)
        self.tf_w = tf.placeholder(dtype=tf.float64)
        self.tf_mul = self.tf_x @ self.tf_w
        self.tf_nonlin = tf.nn.sigmoid(self.tf_mul)

        self.var_x = Variable(name="x")
        self.var_w = Variable(name="w")
        self.var_mul = self.var_x @ self.var_w
        self.var_nonlin = Sigmoid(self.var_mul)

        self.my_input_dict = {self.var_x: self.x, self.var_w: self.w}
        self.tf_input_dict = {self.tf_x: self.x, self.tf_w: self.w}

    def test_f(self):
        graph = self.var_nonlin
        with tf.Session():
            tf_val = self.tf_nonlin.eval(self.tf_input_dict)

        my_val = graph.eval(self.my_input_dict)

        np.testing.assert_allclose(my_val, tf_val)

    def test_oneop_df_w(self):
        graph = self.var_mul
        with tf.Session():
            tf_grads = tf.gradients(self.tf_mul, self.tf_w)[0].eval(self.tf_input_dict)

        grad_ops = Grad(graph, self.var_w)
        my_grads = grad_ops.eval(self.my_input_dict)

        np.testing.assert_allclose(my_grads + self.w, tf_grads + self.w)

    def test_oneop_df_x(self):
        graph = self.var_mul
        with tf.Session():
            tf_grads = tf.gradients(self.tf_mul, self.tf_x)[0].eval(self.tf_input_dict)

        grad_ops = Grad(graph, self.var_x)
        my_grads = grad_ops.eval(self.my_input_dict)

        # This is still kind of a dirty hack?
        np.testing.assert_allclose(my_grads + self.x, tf_grads + self.x)

    def test_df_matmul(self):
        graph = self.var_nonlin
        with tf.Session():
            tf_grads = tf.gradients(self.tf_nonlin, self.tf_mul)[0].eval(self.tf_input_dict)

        grad_ops = Grad(graph, wrt=self.var_mul)
        my_grads = grad_ops.eval(self.my_input_dict)

        np.testing.assert_allclose(my_grads, tf_grads)

    def test_df_w(self):
        graph = self.var_nonlin
        with tf.Session():
            tf_grads = tf.gradients(self.tf_nonlin, self.tf_w)[0].eval(self.tf_input_dict)

        grad_ops = Grad(graph, self.var_w)
        my_grads = grad_ops.eval(self.my_input_dict)

        np.testing.assert_allclose(my_grads, tf_grads)

    def test_df_x(self):
        graph = self.var_nonlin

        my_graph, tf_graph = utils.differentiate_n_times(graph, self.tf_nonlin, self.var_x, self.tf_x, n=1)
        with tf.Session():
            if tf_graph is not None:
                tf_grads = tf_graph.eval(feed_dict=self.tf_input_dict)
            else:
                tf_grads = 0
        my_grads = my_graph.eval(self.my_input_dict)

        np.testing.assert_allclose(my_grads, tf_grads)

    def test_df_with_scalar(self):
        graph = 5 * self.var_x
        tf_graph = 5 * self.tf_x
        with tf.Session():
            tf_grads = tf.gradients(tf_graph, self.tf_x)[0].eval(self.tf_input_dict)

        grad_ops = Grad(graph, self.var_x)
        my_grads = grad_ops.eval(self.my_input_dict)

        np.testing.assert_allclose(my_grads, tf_grads)

    def test_df_transpose(self):
        graph = Transpose(self.var_x)
        tf_graph = tf.transpose(self.tf_x)
        with tf.Session():
            tf_grads = tf.gradients(tf_graph, self.tf_x)[0].eval(self.tf_input_dict)

        grad_ops = Grad(graph, wrt=self.var_x)
        my_grads = grad_ops.eval(self.my_input_dict)

        # TODO this might not be such a good idea?
        # Is there a way to dynamically get the shape of the gradient?
        # Here I'm just relying on broadcasting to work
        # But it does seem like the natural way?
        np.testing.assert_allclose(my_grads + self.x, tf_grads + self.x)
