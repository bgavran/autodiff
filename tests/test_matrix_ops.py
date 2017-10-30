from unittest import TestCase

import numpy as np
import tensorflow as tf
import automatic_differentiation as ad

from tests import utils


class TestOneArgOperations(TestCase):
    def setUp(self):
        """
        Just basic testing, needs to be properly expanded

        """
        np.random.seed(1337)
        self.w0_val = np.random.randn(2, 3)
        self.w1_val = np.random.randn(3, 5)
        self.w2_val = np.random.randn(10, 3)
        self.w3_val = np.random.randn(10, 5)
        self.w4_val = np.random.randn(15)

        self.tf_w0 = tf.constant(self.w0_val, dtype=tf.float64)
        self.tf_w1 = tf.constant(self.w1_val, dtype=tf.float64)
        self.tf_w2 = tf.constant(self.w2_val)
        self.tf_w3 = tf.constant(self.w3_val)
        self.tf_w4 = tf.constant(self.w4_val)

        self.my_w0 = ad.Variable(self.w0_val, name="w0_val")
        self.my_w1 = ad.Variable(self.w1_val, name="w1_val")
        self.my_w2 = ad.Variable(self.w2_val, name="w2_val")
        self.my_w3 = ad.Variable(self.w3_val, name="w3_val")
        self.my_w4 = ad.Variable(self.w4_val, name="w4_val")

    def test_softmax_ce_with_logits(self):
        my_graph = ad.SoftmaxCEWithLogits(labels=self.my_w3, logits=ad.Exp(self.my_w3))
        tf_graph = tf.nn.softmax_cross_entropy_with_logits(labels=self.tf_w3, logits=tf.exp(self.tf_w3))
        utils.test_one_op(self, my_graph, tf_graph, [self.my_w3], [self.tf_w3])

    def test_softmax(self):
        # is this a correct test? Should the gradients always be this small?
        my_graph = ad.Softmax(self.my_w4)
        tf_graph = tf.nn.softmax(self.tf_w4)
        utils.test_one_op(self, my_graph, tf_graph, [self.my_w4], [self.tf_w4])

    def test_reshape(self):
        shp = (1, 15)
        my_graph = ad.Reshape(self.my_w1, shp)
        tf_graph = tf.reshape(self.tf_w1, shp)
        utils.test_one_op(self, my_graph, tf_graph, [self.my_w1], [self.tf_w1])

    def test_pad(self):
        pad_val = [[1, 0], [2, 2]]
        constant_values = [0, 0]
        my_graph = ad.Pad(self.my_w0, pad_val, constant_values=constant_values)
        tf_graph = tf.pad(self.tf_w0, pad_val, constant_values=constant_values[0])
        utils.test_one_op(self, my_graph, tf_graph, [self.my_w0], [self.tf_w0])

    def test_slice1(self):
        my_graph = self.my_w1[:, :-1]
        tf_graph = self.tf_w1[:, :-1]
        utils.test_one_op(self, my_graph, tf_graph, [self.my_w1], [self.tf_w1])

    def test_sigmoid(self):
        my_graph = ad.Sigmoid(self.my_w0)
        tf_graph = tf.nn.sigmoid(self.tf_w0)
        utils.test_one_op(self, my_graph, tf_graph, [self.my_w0], [self.tf_w0])

    def test_relu(self):
        my_graph = ad.ReLU(self.my_w0)
        tf_graph = tf.nn.relu(self.tf_w0)
        utils.test_one_op(self, my_graph, tf_graph, [self.my_w0], [self.tf_w0])

    def test_transpose(self):
        my_graph = ad.Transpose(self.my_w0)
        tf_graph = tf.transpose(self.tf_w0)
        utils.test_one_op(self, my_graph, tf_graph, [self.my_w0], [self.tf_w0])

    def test_recipr(self):
        my_graph = ad.Recipr(self.my_w0)
        tf_graph = tf.reciprocal(self.tf_w0)
        utils.test_one_op(self, my_graph, tf_graph, [self.my_w0], [self.tf_w0])

    def test_negate(self):
        my_graph = ad.Negate(self.my_w0)
        tf_graph = tf.negative(self.tf_w0)
        utils.test_one_op(self, my_graph, tf_graph, [self.my_w0], [self.tf_w0])

    def test_log(self):
        my_graph = ad.Log(self.my_w0)
        tf_graph = tf.log(self.tf_w0)
        utils.test_one_op(self, my_graph, tf_graph, [self.my_w0], [self.tf_w0])

    def test_exp(self):
        my_graph = ad.Exp(self.my_w0)
        tf_graph = tf.exp(self.tf_w0)
        utils.test_one_op(self, my_graph, tf_graph, [self.my_w0], [self.tf_w0])

    def test_identity(self):
        my_graph = ad.Identity(self.my_w0)
        tf_graph = tf.identity(self.tf_w0)
        utils.test_one_op(self, my_graph, tf_graph, [self.my_w0], [self.tf_w0])

    def test_tanh(self):
        my_graph = ad.Tanh(self.my_w0)
        tf_graph = tf.tanh(self.tf_w0)
        utils.test_one_op(self, my_graph, tf_graph, [self.my_w0], [self.tf_w0])

    def test_matmul(self):
        my_graph = ad.MatMul(self.my_w0, self.my_w1)
        tf_graph = tf.matmul(self.tf_w0, self.tf_w1)
        utils.test_one_op(self, my_graph, tf_graph, [self.my_w0, self.my_w1], [self.tf_w0, self.tf_w1])

    def test_concat1(self):
        my_graph = ad.Concat(self.my_w0, self.my_w2, axis=0)
        tf_graph = tf.concat([self.tf_w0, self.tf_w2], axis=0)
        utils.test_one_op(self, my_graph, tf_graph, [self.my_w0, self.my_w2], [self.tf_w0, self.tf_w2])

    def test_concat2(self):
        my_graph = ad.Concat(self.my_w2, self.my_w3, axis=1)
        tf_graph = tf.concat([self.tf_w2, self.tf_w3], axis=1)
        utils.test_one_op(self, my_graph, tf_graph, [self.my_w2, self.my_w3], [self.tf_w2, self.tf_w3])

    def test_frobenius_norm(self):
        my_graph = ad.FrobeniusNorm(self.my_w3)
        tf_graph = tf.norm(self.tf_w3)
        utils.test_one_op(self, my_graph, tf_graph, [self.my_w3], [self.tf_w3])
