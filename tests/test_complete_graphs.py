from unittest import TestCase

import numpy as np
import tensorflow as tf
import automatic_differentiation as ad

from tests import utils


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

        self.n_times = 2

    def test_all(self):
        my_vars = [self.my_x, self.my_w1, self.my_w2]
        tf_vars = [self.tf_x, self.tf_w1, self.tf_w2]
        utils.test_one_op(self, self.my_graph, self.tf_graph, my_vars, tf_vars, n=self.n_times)


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
        self.b_val = np.random.randn(1, 5)

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

        self.n_times = 3

    def test_all(self):
        my_vars = [self.my_x, self.my_w, self.my_b, self.var_mul]
        tf_vars = [self.tf_x, self.tf_w, self.tf_b, self.tf_mul]
        utils.test_one_op(self, self.my_graph, self.tf_graph, my_vars, tf_vars, n=self.n_times)


class TestOptimizaton(TestCase):
    def setUp(self):
        np.random.seed(1337)

        batch_size = 100
        input_size = 2
        output_size = 2

        self.x_val = np.random.randn(batch_size, input_size + 1)
        self.w_val = np.random.randn(input_size + 1, output_size) * 0.01
        self.y_val = np.zeros((batch_size, output_size))
        self.y_val[np.random.randint(0, output_size)] = 1

        self.my_x = ad.Variable(self.x_val, name="x")
        self.my_w = ad.Variable(self.w_val, name="w")
        self.my_y = ad.Variable(self.y_val, name="y")
        self.my_mul = self.my_x @ self.my_w
        self.my_softmax_ce_logits = ad.SoftmaxCEWithLogits(labels=self.my_y, logits=self.my_mul)
        self.my_grad_w = ad.grad(self.my_softmax_ce_logits, [self.my_w])[0]

        self.tf_x = tf.constant(self.x_val, dtype=tf.float64)
        self.tf_w = tf.constant(self.w_val, dtype=tf.float64)
        self.tf_y = tf.constant(self.y_val, dtype=tf.float64)
        self.tf_mul = self.tf_x @ self.tf_w
        self.tf_softmax_ce_logits = tf.nn.softmax_cross_entropy_with_logits(labels=self.tf_y, logits=self.tf_mul)
        self.tf_grad_w = tf.gradients(self.tf_softmax_ce_logits, self.tf_w)[0]

        self.n_times = 1

    def test_all(self):
        my_vars = [self.my_x, self.my_w, self.my_mul, self.my_softmax_ce_logits]
        tf_vars = [self.tf_x, self.tf_w, self.tf_mul, self.tf_softmax_ce_logits]
        utils.test_one_op(self, self.my_grad_w, self.tf_grad_w, my_vars, tf_vars, n=self.n_times)
