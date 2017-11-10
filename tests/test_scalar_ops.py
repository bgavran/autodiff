from unittest import TestCase

import numpy as np
import tensorflow as tf
import automatic_differentiation as ad

from tests import utils, numerical_check


class TestTwoArgScalarOperations(TestCase):
    def setUp(self):
        np.random.seed(1336)
        self.w0_val = np.random.rand()
        self.w1_val = np.random.rand()

        self.tf_w0 = tf.constant(self.w0_val, dtype=tf.float64)
        self.tf_w1 = tf.constant(self.w1_val, dtype=tf.float64)

        self.my_w0 = ad.Variable(self.w0_val, name="w0_val")
        self.my_w1 = ad.Variable(self.w1_val, name="w1_val")

    def custom_test(self, my_graph, tf_graph, my_vars, tf_vars):
        utils.custom_test(self, my_graph, my_vars, tf_graph, tf_vars)

    def test_add(self):
        my_graph = ad.Add(self.my_w0, self.my_w1)
        tf_graph = tf.add(self.tf_w0, self.tf_w1)
        my_vars = [self.my_w0, self.my_w1]
        tf_vars = [self.tf_w0, self.tf_w1]
        self.custom_test(my_graph, tf_graph, my_vars, tf_vars)

    def test_mul(self):
        my_graph = ad.Mul(self.my_w0, self.my_w1)
        tf_graph = tf.multiply(self.tf_w0, self.tf_w1)
        my_vars = [self.my_w0, self.my_w1]
        tf_vars = [self.tf_w0, self.tf_w1]
        self.custom_test(my_graph, tf_graph, my_vars, tf_vars)

    def test_power(self):
        my_graph = ad.Pow(self.my_w0, self.my_w1)
        tf_graph = tf.pow(self.tf_w0, self.tf_w1)
        my_vars = [self.my_w0, self.my_w1]
        tf_vars = [self.tf_w0, self.tf_w1]
        self.custom_test(my_graph, tf_graph, my_vars, tf_vars)

    def test_squared_difference(self):
        my_graph = ad.SquaredDifference(self.my_w0, self.my_w1)
        tf_graph = tf.squared_difference(self.tf_w0, self.tf_w1)
        my_vars = [self.my_w0, self.my_w1]
        tf_vars = [self.tf_w0, self.tf_w1]
        self.custom_test(my_graph, tf_graph, my_vars, tf_vars)
