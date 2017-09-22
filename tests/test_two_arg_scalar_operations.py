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

        self.tf_w0 = tf.placeholder(dtype=tf.float64)
        self.tf_w1 = tf.placeholder(dtype=tf.float64)

        self.my_w0 = Variable(self.w0_val, name="w0_val")
        self.my_w1 = Variable(self.w1_val, name="w1_val")

        self.tf_input_dict = {self.tf_w0: self.w0_val, self.tf_w1: self.w1_val}

    def oneop_df_n_times(self, my_op, tf_op, wrt_vars, n=1):
        my_var, tf_var = wrt_vars

        tf_graph = tf_op(self.tf_w0, self.tf_w1)
        my_graph = my_op(self.my_w0, self.my_w1)

        my_graph, tf_graph = utils.differentiate_n_times(my_graph, tf_graph, my_var, tf_var, n=n)

        with tf.Session():
            if tf_graph is not None:
                tf_grads = tf_graph.eval(feed_dict=self.tf_input_dict)
            else:
                tf_grads = 0
        my_grads = my_graph.eval()

        print("---------- " + str(n) + "df ----------")
        print("My_val:", my_grads)
        print("Tf_val:", tf_grads)
        my_val = my_grads + self.w0_val
        tf_val = tf_grads + self.w0_val
        np.testing.assert_allclose(my_val, tf_val)

    @timeout_decorator.timeout(2)
    def oneop(self, my_op, tf_op):
        print("---------- " + "inputs" + "   ----------")
        print("w0_val:", self.w0_val)
        print("w1_val:", self.w1_val)
        print("-------------------------")
        with self.subTest("f"):
            self.oneop_df_n_times(my_op, tf_op, [self.my_w0, self.tf_w0], n=0)
        with self.subTest("df_wrt_w0"):
            self.oneop_df_n_times(my_op, tf_op, [self.my_w0, self.tf_w0], n=1)
        with self.subTest("df_wrt_w1"):
            self.oneop_df_n_times(my_op, tf_op, [self.my_w1, self.tf_w1], n=1)

        with self.subTest("2df_wrt_w0"):
            self.oneop_df_n_times(my_op, tf_op, [self.my_w0, self.tf_w0], n=2)
        with self.subTest("2df_wrt_w1"):
            self.oneop_df_n_times(my_op, tf_op, [self.my_w1, self.tf_w1], n=2)

        with self.subTest("3df_wrt_w0"):
            self.oneop_df_n_times(my_op, tf_op, [self.my_w0, self.tf_w0], n=3)
        with self.subTest("3df_wrt_w1"):
            self.oneop_df_n_times(my_op, tf_op, [self.my_w1, self.tf_w1], n=3)

    def test_add(self):
        self.oneop(Add, tf.add)

    def test_mul(self):
        self.oneop(Mul, tf.multiply)

    def test_power(self):
        self.oneop(Pow, tf.pow)

    def test_squared_difference(self):
        self.oneop(SquaredDifference, tf.squared_difference)
