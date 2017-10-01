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

    @timeout_decorator.timeout(2)
    def oneop(self, var_op, tf_op):
        print("---------- " + "inputs" + "   ----------")
        print("w0_val:", self.w0_val)
        print("w1_val:", self.w1_val)
        print("-------------------------")

        my_graph = var_op(self.my_w0, self.my_w1)
        tf_graph = tf_op(self.tf_w0, self.tf_w1)

        with self.subTest("f"):
            utils.oneop_f(my_graph, tf_graph)
        with self.subTest("df_wrt_w0"):
            utils.oneop_df_n_times(self, my_graph, tf_graph, [self.my_w0, self.tf_w0], n=1)
        with self.subTest("df_wrt_w1"):
            utils.oneop_df_n_times(self, my_graph, tf_graph, [self.my_w1, self.tf_w1], n=1)

        with self.subTest("2df_wrt_w0"):
            utils.oneop_df_n_times(self, my_graph, tf_graph, [self.my_w0, self.tf_w0], n=2)
        with self.subTest("2df_wrt_w1"):
            utils.oneop_df_n_times(self, my_graph, tf_graph, [self.my_w1, self.tf_w1], n=2)

        with self.subTest("3df_wrt_w0"):
            utils.oneop_df_n_times(self, my_graph, tf_graph, [self.my_w0, self.tf_w0], n=3)
        with self.subTest("3df_wrt_w1"):
            utils.oneop_df_n_times(self, my_graph, tf_graph, [self.my_w1, self.tf_w1], n=3)

    def test_matmul(self):
        self.oneop(MatMul, tf.matmul)
