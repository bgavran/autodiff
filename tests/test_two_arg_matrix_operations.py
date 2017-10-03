from unittest import TestCase
import timeout_decorator

import tensorflow as tf

from core.ops import *
from tests import utils


class TestTwoArgScalarOperations(TestCase):
    def setUp(self):
        np.random.seed(1337)

        self.w0_val = np.random.randn(2, 3)
        self.w1_val = np.random.randn(3, 5)

        self.tf_w0 = tf.constant(self.w0_val)
        self.tf_w1 = tf.constant(self.w1_val)

        self.my_w0 = Variable(self.w0_val, name="w0_val")
        self.my_w1 = Variable(self.w1_val, name="w1_val")

        self.n_times = 3

    def test_matmul(self):
        my_graph = MatMul(self.my_w0, self.my_w1)
        tf_graph = tf.matmul(self.tf_w0, self.tf_w1)
        utils.test_one_op(self, my_graph, tf_graph, [self.my_w0, self.my_w1], [self.tf_w0, self.tf_w1], n=self.n_times)
