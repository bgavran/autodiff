from unittest import TestCase

import numpy as np
import tensorflow as tf

import autodiff as ad
import utils


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

        self.my_x = ad.Variable(self.x_val, name="x_val")
        self.my_w1 = ad.Variable(self.w1_val, name="w1_val")
        self.my_w2 = ad.Variable(self.w2_val, name="w2_val")
        self.var_h = ad.Sigmoid(self.my_x @ self.my_w1)
        self.var_o = ad.Sigmoid(self.var_h @ self.my_w2)

        self.my_graph = self.var_o
        self.tf_graph = self.tf_o

    def test_all(self):
        my_vars = [self.my_x, self.my_w1, self.my_w2]
        tf_vars = [self.tf_x, self.tf_w1, self.tf_w2]
        utils.custom_test(self, self.my_graph, my_vars, self.tf_graph, tf_vars)


class TestBroadcastElementwiseGrad(TestCase):
    def setUp(self):
        np.random.seed(1337)
        h_val = np.random.randn(2, 5)
        b0_val = np.random.randn(5)
        b1_val = np.random.randn(1, 5)
        b2_val = 7

        self.my_h = ad.Variable(h_val, name="h")
        self.my_b0 = ad.Variable(b0_val, name="b0")
        self.my_b1 = ad.Variable(b1_val, name="b1")
        self.my_b2 = ad.Variable(b2_val, name="b2")

        self.tf_h = tf.constant(h_val, dtype=tf.float64)
        self.tf_b0 = tf.constant(b0_val, dtype=tf.float64)
        self.tf_b1 = tf.constant(b1_val, dtype=tf.float64)
        self.tf_b2 = tf.constant(b2_val, dtype=tf.float64)

    def test_sum(self):
        my_graph = self.my_h + self.my_b0 + self.my_b1 + self.my_b2
        tf_graph = self.tf_h + self.tf_b0 + self.tf_b1 + self.tf_b2
        wrt_vars = [self.my_b0, self.my_b1, self.my_b2]
        tf_vars = [self.tf_b0, self.tf_b1, self.tf_b2]
        utils.custom_test(self, my_graph, wrt_vars, tf_graph, tf_vars)

    def test_mul(self):
        my_graph = self.my_h * self.my_b0 * self.my_b1 + self.my_b2
        tf_graph = self.tf_h * self.tf_b0 * self.tf_b1 + self.tf_b2
        wrt_vars = [self.my_b0, self.my_b1, self.my_b2]
        tf_vars = [self.tf_b0, self.tf_b1, self.tf_b2]
        utils.custom_test(self, my_graph, wrt_vars, tf_graph, tf_vars)


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
        self.b_val = np.random.randn(5)

        self.tf_x = tf.constant(self.x_val, dtype=tf.float64)
        self.tf_w = tf.constant(self.w_val, dtype=tf.float64)
        self.tf_b = tf.constant(self.b_val, dtype=tf.float64)
        self.tf_mul = self.tf_x @ self.tf_w + self.tf_b
        self.tf_graph = tf.nn.sigmoid(self.tf_mul)

        self.my_x = ad.Variable(self.x_val, name="x_val")
        self.my_w = ad.Variable(self.w_val, name="w_val")
        self.my_b = ad.Variable(self.b_val, name="b_val")

        self.var_mul = self.my_x @ self.my_w + self.my_b
        self.my_graph = ad.Sigmoid(self.var_mul)

    def test_all(self):
        my_vars = [self.my_x, self.my_w, self.my_b, self.var_mul]
        tf_vars = [self.tf_x, self.tf_w, self.tf_b, self.tf_mul]
        utils.custom_test(self, self.my_graph, my_vars, self.tf_graph, tf_vars)
