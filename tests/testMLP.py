from unittest import TestCase

import tensorflow as tf

from core.ops import *


class TestMLP(TestCase):
    def setUp(self):
        """
        Creating a multi-layer perceptron with one hidden layer

        """
        np.random.seed(1337)

        batch_size = 16
        input_size = 20
        hidden_size = 40
        output_size = 5
        self.x = np.random.rand(batch_size, input_size)
        self.w1 = np.random.rand(input_size, hidden_size)
        self.w2 = np.random.rand(hidden_size, output_size)

        self.tf_x = tf.constant(self.x)
        self.tf_w1 = tf.constant(self.w1)
        self.tf_w2 = tf.constant(self.w2)
        self.tf_h = tf.nn.sigmoid(self.tf_x @ self.tf_w1)
        self.tf_o = tf.nn.sigmoid(self.tf_h @ self.tf_w2)

        self.var_x = Variable(name="x")
        self.var_w1 = Variable(name="w1")
        self.var_w2 = Variable(name="w2")
        self.var_h = Sigmoid(self.var_x @ self.var_w1)
        self.var_o = Sigmoid(self.var_h @ self.var_w2)

        self.input_dict = {"x": self.x, "w1": self.w1, "w2": self.w2}
        self.graph = self.var_o
        self.tf_graph = self.tf_o

    def test_f(self):
        with tf.Session():
            tf_val = self.tf_graph.eval()

        my_val = self.graph.f(self.input_dict)

        np.testing.assert_allclose(my_val, tf_val)

    def test_df_x(self):
        with tf.Session():
            tf_grads = tf.gradients(self.tf_graph, self.tf_x)[0].eval()

        self.graph.compute_derivatives(self.input_dict)
        my_grads = self.graph.accumulate_all_gradients(wrt="x")

        np.testing.assert_allclose(my_grads, tf_grads)

    def test_df_w1(self):
        with tf.Session():
            tf_grads = tf.gradients(self.tf_graph, self.tf_w1)[0].eval()

        self.graph.compute_derivatives(self.input_dict)
        my_grads = self.graph.accumulate_all_gradients(wrt="w1")

        np.testing.assert_allclose(my_grads, tf_grads)

    def test_df_w2(self):
        with tf.Session():
            tf_grads = tf.gradients(self.tf_graph, self.tf_w2)[0].eval()

        self.graph.compute_derivatives(self.input_dict)
        my_grads = self.graph.accumulate_all_gradients(wrt="w2")

        np.testing.assert_allclose(my_grads, tf_grads)
