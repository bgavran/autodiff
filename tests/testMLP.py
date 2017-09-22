from unittest import TestCase

import tensorflow as tf

from core.ops import *


class TestMLP(TestCase):
    def setUp(self):
        """
        Creating true multi-layer perceptron with one hidden layer

        """
        np.random.seed(1337)

        batch_size = 16
        input_size = 20
        hidden_size = 40
        output_size = 5
        self.x_val = np.random.randn(batch_size, input_size)
        self.w1_val = np.random.randn(input_size, hidden_size)
        self.w2_val = np.random.randn(hidden_size, output_size)

        self.tf_x = tf.constant(self.x_val)
        self.tf_w1 = tf.constant(self.w1_val)
        self.tf_w2 = tf.constant(self.w2_val)
        self.tf_h = tf.nn.sigmoid(self.tf_x @ self.tf_w1)
        self.tf_o = tf.nn.sigmoid(self.tf_h @ self.tf_w2)

        self.my_x = Variable(self.x_val, name="x_val")
        self.my_w1 = Variable(self.w1_val, name="w1_val")
        self.my_w2 = Variable(self.w2_val, name="w2_val")
        self.var_h = Sigmoid(self.my_x @ self.my_w1)
        self.var_o = Sigmoid(self.var_h @ self.my_w2)

        self.graph = self.var_o
        self.tf_graph = self.tf_o

    def test_f(self):
        with tf.Session():
            tf_val = self.tf_graph.eval()

        my_val = self.graph.eval()

        np.testing.assert_allclose(my_val, tf_val)

    def test_df_x(self):
        with tf.Session():
            tf_grads = tf.gradients(self.tf_graph, self.tf_x)[0].eval()

        grad_ops = Grad(self.graph, self.my_x)
        my_grads = grad_ops.eval()

        np.testing.assert_allclose(my_grads, tf_grads)

    def test_df_w1(self):
        with tf.Session():
            tf_grads = tf.gradients(self.tf_graph, self.tf_w1)[0].eval()

        grad_ops = Grad(self.graph, self.my_w1)
        my_grads = grad_ops.eval()

        np.testing.assert_allclose(my_grads, tf_grads)

    def test_df_w2(self):
        with tf.Session():
            tf_grads = tf.gradients(self.tf_graph, self.tf_w2)[0].eval()

        grad_ops = Grad(self.graph, self.my_w2)
        my_grads = grad_ops.eval()

        np.testing.assert_allclose(my_grads, tf_grads)
