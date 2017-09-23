from unittest import TestCase

import tensorflow as tf

from core.ops import *


class TestEinSum(TestCase):
    def setUp(self):
        np.random.seed(1337)
        self.w0_val = np.random.randn(2, 3)
        self.w1_val = np.random.randn(3, 5)
        self.w2_val = np.random.randn(5, 5)
        self.w3_val = np.random.randn(5, 7)

        self.tf_w0 = tf.constant(self.w0_val)
        self.tf_w1 = tf.constant(self.w1_val)
        self.tf_w2 = tf.constant(self.w2_val)
        self.tf_w3 = tf.constant(self.w3_val)

        self.my_w0 = Variable(self.w0_val, name="w0_val")
        self.my_w1 = Variable(self.w1_val, name="w1_val")
        self.my_w2 = Variable(self.w2_val, name="w2_val")
        self.my_w3 = Variable(self.w3_val, name="w3_val")

    def test_twovars_f(self):
        tf_graph = tf.einsum("dt,tp->dp", self.tf_w0, self.tf_w1)
        graph = EinSum("dt,tp->dp", self.my_w0, self.my_w1)
        with tf.Session():
            tf_val = tf_graph.eval()

        my_val = graph.f()

        np.testing.assert_allclose(my_val, tf_val)

    def test_threevars_f(self):
        tf_graph = tf.einsum("dt,tp,pr->dtp", self.tf_w0, self.tf_w1, self.tf_w2)
        graph = EinSum("dt,tp,pr->dtp", self.my_w0, self.my_w1, self.my_w2)
        with tf.Session():
            tf_val = tf_graph.eval()

        my_val = graph.f()

        np.testing.assert_allclose(my_val, tf_val)

        # def test_threevars_df_third(self):
        #     tf_graph = tf.einsum("dt,tp,pr->dtp", self.tf_w0, self.tf_w1, self.tf_w2)
        #     graph = EinSum("dt,tp,pr->dtp", self.my_w0, self.my_w1, self.my_w2)
        #     with tf.Session():
        #         tf_grads = tf.gradients(tf_graph, self.tf_w2)[0].eval()
        #     print(tf_grads)
        #
        #     graph.compute_derivatives(self.my_input_dict)
        #     my_grads = graph.accumulate_all_gradients(wrt="w2_val")
        #
        #     print(my_grads)
        #     np.testing.assert_allclose(my_grads, tf_grads)
