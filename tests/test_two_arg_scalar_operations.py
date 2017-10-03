from unittest import TestCase
import timeout_decorator

import tensorflow as tf

from core.ops import *
from tests import utils


class TestTwoArgScalarOperations(TestCase):
    def setUp(self):
        np.random.seed(1336)
        self.w0_val = np.random.rand()
        self.w1_val = np.random.rand()

        self.tf_w0 = tf.constant(self.w0_val, dtype=tf.float64)
        self.tf_w1 = tf.constant(self.w1_val, dtype=tf.float64)

        self.my_w0 = Variable(self.w0_val, name="w0_val")
        self.my_w1 = Variable(self.w1_val, name="w1_val")

        self.n_times = 3

    def test_add(self):
        my_graph = Add(self.my_w0, self.my_w1)
        tf_graph = tf.add(self.tf_w0, self.tf_w1)
        utils.test_one_op(self, my_graph, tf_graph, [self.my_w0, self.my_w1], [self.tf_w0, self.tf_w1], n=self.n_times)

    def test_mul(self):
        my_graph = Mul(self.my_w0, self.my_w1)
        tf_graph = tf.multiply(self.tf_w0, self.tf_w1)
        utils.test_one_op(self, my_graph, tf_graph, [self.my_w0, self.my_w1], [self.tf_w0, self.tf_w1], n=self.n_times)

    def test_power(self):
        my_graph = Pow(self.my_w0, self.my_w1)
        tf_graph = tf.pow(self.tf_w0, self.tf_w1)
        utils.test_one_op(self, my_graph, tf_graph, [self.my_w0, self.my_w1], [self.tf_w0, self.tf_w1], n=self.n_times)

    def test_squared_difference(self):
        my_graph = SquaredDifference(self.my_w0, self.my_w1)
        tf_graph = tf.squared_difference(self.tf_w0, self.tf_w1)
        utils.test_one_op(self, my_graph, tf_graph, [self.my_w0, self.my_w1], [self.tf_w0, self.tf_w1], n=self.n_times)
