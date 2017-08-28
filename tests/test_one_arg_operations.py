from unittest import TestCase

import tensorflow as tf

from core.ops import *


class TestOneArgOperations(TestCase):
    def setUp(self):
        np.random.seed(1337)
        self.w0 = np.random.rand(2, 3)

        self.tf_w0 = tf.constant(self.w0)
        self.var_w0 = Variable(name="w0")

        self.input_dict = {self.var_w0: self.w0}

    def oneop_f(self, var_op, tf_op):
        tf_graph = tf_op(self.tf_w0)
        graph = var_op(self.var_w0)

        with tf.Session():
            tf_val = tf_graph.eval()

        my_val = graph.eval(self.input_dict)

        print("---------- input ----------")
        print("w0:", self.w0)
        print("---------------------------")

        print("---------- f ----------")
        print("My_val:", my_val)
        print("Tf_val:", tf_val)
        print("-----------------------")
        np.testing.assert_allclose(my_val, tf_val)

    def oneop_df_n_times(self, var_op, tf_op, n=1):
        tf_graph = tf_op(self.tf_w0)
        graph = var_op(self.var_w0)
        with tf.Session():
            for _ in range(n):
                graph = Grad(graph, wrt=self.var_w0)
                if tf_graph is not None:
                    tf_graph = tf.gradients(tf_graph, self.tf_w0)[0]
            if tf_graph is not None:
                tf_grads = tf_graph.eval()
            else:
                tf_grads = 0

        my_grads = graph.eval(self.input_dict)

        print("---------- " + str(n) + "df ----------")
        print("My_val:", my_grads)
        print("Tf_val:", tf_grads)
        print("-------------------------")
        my_val = my_grads + self.w0
        tf_val = tf_grads + self.w0
        np.testing.assert_allclose(my_val, tf_val)

    def oneop(self, var_op, tf_op):
        with self.subTest("f"):
            self.oneop_f(var_op, tf_op)
        with self.subTest("df"):
            self.oneop_df_n_times(var_op, tf_op, n=1)
        with self.subTest("2df"):
            self.oneop_df_n_times(var_op, tf_op, n=2)
        with self.subTest("3df"):
            self.oneop_df_n_times(var_op, tf_op, n=3)
        with self.subTest("4df"):
            self.oneop_df_n_times(var_op, tf_op, n=4)

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
