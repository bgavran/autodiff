from unittest import TestCase
import timeout_decorator

import numpy as np
import tensorflow as tf
import automatic_differentiation as ad

from tests import utils


class TestTwoArgScalarOperations(TestCase):
    def setUp(self):
        np.random.seed(1337)

        self.w0_val = np.random.randn(2, 3)
        self.w1_val = np.random.randn(3, 5)
        self.w2_val = np.random.randn(10, 3)
        self.w3_val = np.random.randn(10, 5)

        self.tf_w0 = tf.constant(self.w0_val)
        self.tf_w1 = tf.constant(self.w1_val)
        self.tf_w2 = tf.constant(self.w2_val)
        self.tf_w3 = tf.constant(self.w3_val)

        self.my_w0 = ad.Variable(self.w0_val, name="w0_val")
        self.my_w1 = ad.Variable(self.w1_val, name="w1_val")
        self.my_w2 = ad.Variable(self.w2_val, name="w2_val")
        self.my_w3 = ad.Variable(self.w3_val, name="w3_val")

        self.n_times = 3

    def test_matmul(self):
        my_graph = ad.MatMul(self.my_w0, self.my_w1)
        tf_graph = tf.matmul(self.tf_w0, self.tf_w1)
        utils.test_one_op(self, my_graph, tf_graph, [self.my_w0, self.my_w1], [self.tf_w0, self.tf_w1], n=self.n_times)

    def test_concat1(self):
        my_graph = ad.Concat(self.my_w0, self.my_w2, axis=0)
        tf_graph = tf.concat([self.tf_w0, self.tf_w2], axis=0)
        utils.test_one_op(self, my_graph, tf_graph, [self.my_w0, self.my_w2], [self.tf_w0, self.tf_w2], n=self.n_times)

    def test_concat2(self):
        my_graph = ad.Concat(self.my_w2, self.my_w3, axis=1)
        tf_graph = tf.concat([self.tf_w2, self.tf_w3], axis=1)
        utils.test_one_op(self, my_graph, tf_graph, [self.my_w2, self.my_w3], [self.tf_w2, self.tf_w3], n=self.n_times)

    def test_frobenius_norm(self):
        my_graph = ad.FrobeniusNorm(self.my_w3)
        tf_graph = tf.norm(self.tf_w3)
        utils.test_one_op(self, my_graph, tf_graph, [self.my_w3], [self.tf_w3], n=self.n_times)
