from unittest import TestCase

import tensorflow as tf

from core.ops import *


class TestEinSum(TestCase):
    def setUp(self):
        np.random.seed(1337)
        self.w0 = np.random.rand(2, 3)
        self.w1 = np.random.rand(3, 5)
        self.w2 = np.random.rand(5, 5)
        self.w3 = np.random.rand(5, 7)

        self.tf_w0 = tf.constant(self.w0)
        self.tf_w1 = tf.constant(self.w1)
        self.tf_w2 = tf.constant(self.w2)
        self.tf_w3 = tf.constant(self.w3)

        self.var_w0 = Variable(name="w0")
        self.var_w1 = Variable(name="w1")
        self.var_w2 = Variable(name="w2")
        self.var_w3 = Variable(name="w3")

        self.input_dict = {"w0": self.w0,
                           "w1": self.w1,
                           "w2": self.w2,
                           "w3": self.w3}

    def test_twovars_f(self):
        tf_graph = tf.einsum("dt,tp->dp", self.tf_w0, self.tf_w1)
        graph = EinSum("dt,tp->dp", self.var_w0, self.var_w1)
        with tf.Session():
            tf_val = tf_graph.eval()

        my_val = graph.eval(self.input_dict)

        np.testing.assert_allclose(my_val, tf_val)

    def test_threevars_f(self):
        tf_graph = tf.einsum("dt,tp,pr->dtp", self.tf_w0, self.tf_w1, self.tf_w2)
        graph = EinSum("dt,tp,pr->dtp", self.var_w0, self.var_w1, self.var_w2)
        with tf.Session():
            tf_val = tf_graph.eval()

        my_val = graph.eval(self.input_dict)

        np.testing.assert_allclose(my_val, tf_val)

    # def test_threevars_df_third(self):
    #     tf_graph = tf.einsum("dt,tp,pr->dtp", self.tf_w0, self.tf_w1, self.tf_w2)
    #     graph = EinSum("dt,tp,pr->dtp", self.var_w0, self.var_w1, self.var_w2)
    #     with tf.Session():
    #         tf_grads = tf.gradients(tf_graph, self.tf_w2)[0].eval()
    #     print(tf_grads)
    #
    #     graph.compute_derivatives(self.input_dict)
    #     my_grads = graph.accumulate_all_gradients(wrt="w2")
    #
    #     print(my_grads)
    #     np.testing.assert_allclose(my_grads, tf_grads)

