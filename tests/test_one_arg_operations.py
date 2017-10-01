from unittest import TestCase
import timeout_decorator

import tensorflow as tf

from core.ops import *

from tests import utils


class TestOneArgOperations(TestCase):
    def setUp(self):
        np.random.seed(1337)
        self.w0_val = np.random.randn(2, 3)

        self.tf_w0 = tf.placeholder(dtype=tf.float64)
        self.my_w0 = Variable(self.w0_val, name="w0_val")

        self.tf_input_dict = {self.tf_w0: self.w0_val}

    def oneop_f(self, var_op, tf_op, **kwargs):
        tf_graph = tf_op(self.tf_w0, **kwargs)
        graph = var_op(self.my_w0, **kwargs)

        with tf.Session():
            tf_val = tf_graph.eval(self.tf_input_dict)

        my_val = graph.eval()

        print("---------- input ----------")
        print("w0_val:", self.w0_val)
        print("---------------------------")

        print("---------- f ----------")
        print("My_val:", my_val)
        print("Tf_val:", tf_val)
        print("-----------------------")
        np.testing.assert_allclose(my_val, tf_val)

    def oneop_df_n_times(self, var_op, tf_op, n=1, **kwargs):
        my_var, tf_var = self.my_w0, self.tf_w0

        tf_graph = tf_op(self.tf_w0, **kwargs)
        my_graph = var_op(self.my_w0, **kwargs)

        my_graph, tf_graph = utils.differentiate_n_times(my_graph, tf_graph, my_var, tf_var, n=n)

        with tf.Session():
            if tf_graph is not None:
                tf_grads = tf_graph.eval(self.tf_input_dict)
            else:
                tf_grads = 0
        my_grads = my_graph.eval()

        print("---------- " + str(n) + "df ----------")
        print("My_val:", my_grads)
        print("Tf_val:", tf_grads)
        print("-------------------------")
        my_val = my_grads + self.w0_val
        tf_val = tf_grads + self.w0_val
        np.testing.assert_allclose(my_val, tf_val)

    @timeout_decorator.timeout(1)
    def oneop(self, var_op, tf_op, **kwargs):
        with self.subTest("f"):
            self.oneop_f(var_op, tf_op, **kwargs)
        with self.subTest("df"):
            self.oneop_df_n_times(var_op, tf_op, n=1, **kwargs)
        with self.subTest("2df"):
            self.oneop_df_n_times(var_op, tf_op, n=2, **kwargs)
        with self.subTest("3df"):
            self.oneop_df_n_times(var_op, tf_op, n=3, **kwargs)
        # with self.subTest("4df"):
        #     self.oneop_df_n_times(var_op, tf_op, n=4, **kwargs)

    # def test_reshape(self):
    #     self.oneop(ReshapeLike, tf.reshape, shape=(1, 6))

    def test_sigmoid(self):
        self.oneop(Sigmoid, tf.nn.sigmoid)

    def test_relu(self):
        self.oneop(ReLU, tf.nn.relu)

    def test_transpose(self):
        self.oneop(Transpose, tf.transpose)

    def test_recipr(self):
        self.oneop(Recipr, tf.reciprocal)

    def test_negate(self):
        self.oneop(Negate, tf.negative)

    def test_log(self):
        self.oneop(Log, tf.log)

    def test_exp(self):
        self.oneop(Exp, tf.exp)

    def test_identity(self):
        self.oneop(Identity, tf.identity)

    def test_tanh(self):
        self.oneop(Tanh, tf.tanh)
