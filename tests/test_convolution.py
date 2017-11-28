from unittest import TestCase

import numpy as np
import tensorflow as tf
import autodiff as ad

import utils


class TestOneArgOperations(TestCase):
    def setUp(self):
        """
        Just basic testing, needs to be properly expanded

        """
        np.random.seed(1337)
        self.w0_val = np.random.rand(10, 28, 28, 3)
        num_filters = 15
        self.w1_val = np.random.rand(1, 5, 5, 3, num_filters)

        self.my_w0 = ad.Variable(self.w0_val, name="w0_val")
        self.my_w1 = ad.Variable(self.w1_val, name="w1_val")

        self.tf_w0 = tf.constant(self.w0_val, dtype=tf.float32)
        self.tf_w1 = tf.constant(np.transpose(self.w1_val, [1, 2, 3, 4, 0])[:, :, :, :, 0], dtype=tf.float32)

        # def test_conv1(self):
        #     my_graph = ad.Einsum("nhwoc->nhwc", ad.Convolution(self.my_w0, self.my_w1))
        #     wrt_vars = [self.my_w0, self.my_w1]
        #
        #     tf_graph = tf.nn.convolution(self.tf_w0, self.tf_w1, padding="VALID")
        #     tf_wrt_vars = [self.tf_w0, self.tf_w1]
        #     utils.custom_test(self, my_graph, wrt_vars, tf_graph=tf_graph, tf_wrt_vars=tf_wrt_vars)
        #
        # def test_conv2(self):
        #     c = 3
        #     x_val = np.random.randn(3, c, 10, 10)
        #     num_filters = 4
        #     kernel_size = 5
        #     w_val = np.random.randn(num_filters, kernel_size ** 2 * c)
        #
        #     x = ad.Variable(x_val, name="x")
        #     w = ad.Variable(w_val, name="w")
        #
        #     my_graph = ad.Convolution(x, w)
        #     wrt_vars = [x, w]
        #     utils.custom_test(self, my_graph=my_graph, my_wrt_vars=wrt_vars)
