from unittest import TestCase

import numpy as np
import tensorflow as tf
import automatic_differentiation as ad

from tests import utils


class TestEinSum(TestCase):
    def setUp(self):
        np.random.seed(1337)
        self.w0_val = np.random.randn(2, 3)
        self.w1_val = np.random.randn(3, 5)
        self.w2_val = np.random.randn(5, 5)
        self.w3_val = np.random.randn(2, 3, 5, 7)

        self.tf_w0 = tf.constant(self.w0_val)
        self.tf_w1 = tf.constant(self.w1_val)
        self.tf_w2 = tf.constant(self.w2_val)
        self.tf_w3 = tf.constant(self.w3_val)

        self.my_w0 = ad.Variable(self.w0_val, name="w0_val")
        self.my_w1 = ad.Variable(self.w1_val, name="w1_val")
        self.my_w2 = ad.Variable(self.w2_val, name="w2_val")
        self.my_w3 = ad.Variable(self.w3_val, name="w3_val")

        # TODO figure out what works and what doesn't
        """
        Doesn't work: when the right side has less letters than left one. Two possible ways it can happen: 
        * if we're summing over one of the variables (letters) and thus don't put it on the right side
        * if one op has an "extra" letter which gets summed over (it might be the same thing as above)
        # Update: the above should be fixed? Need to thoroughly check this
        
        Doesn't work: ellipsis
        
        Works: normal multiplication of many variables where the w.r.t. variable doesn't have any of its axes summed
        """

    def custom_einsum_f(self, op_str, my_args, tf_args, my_wrt, tf_wrt):
        tf_graph = tf.einsum(op_str, *tf_args)
        my_graph = ad.Einsum(op_str, *my_args)

        utils.test_one_op(self, my_graph, tf_graph, my_wrt, tf_wrt)

    def test_onearg_identity(self):
        my_args = [self.my_w3]
        tf_args = [self.tf_w3]
        op_str = "ijkl->ijkl"
        self.custom_einsum_f(op_str, my_args, tf_args, my_wrt=my_args, tf_wrt=tf_args)

    def test_onearg_sum1(self):
        my_args = [self.my_w3]
        tf_args = [self.tf_w3]
        op_str = "ijkl->ijk"
        self.custom_einsum_f(op_str, my_args, tf_args, my_wrt=my_args, tf_wrt=tf_args)

    def test_onearg_sum2(self):
        my_args = [self.my_w3]
        tf_args = [self.tf_w3]
        op_str = "ijkl->ij"
        self.custom_einsum_f(op_str, my_args, tf_args, my_wrt=my_args, tf_wrt=tf_args)

    def test_onearg_sum3(self):
        my_args = [self.my_w3]
        tf_args = [self.tf_w3]
        op_str = "ijkl->i"
        self.custom_einsum_f(op_str, my_args, tf_args, my_wrt=my_args, tf_wrt=tf_args)

    def test_onearg_sum4(self):
        my_args = [self.my_w3]
        tf_args = [self.tf_w3]
        op_str = "ijkl->i"
        self.custom_einsum_f(op_str, my_args, tf_args, my_wrt=my_args, tf_wrt=tf_args)

    def test_onearg_sum5(self):
        my_args = [self.my_w3]
        tf_args = [self.tf_w3]
        op_str = "ijkl->"
        self.custom_einsum_f(op_str, my_args, tf_args, my_wrt=my_args, tf_wrt=tf_args)

    def test_twovars_f(self):
        my_args = [self.my_w0, self.my_w1]
        tf_args = [self.tf_w0, self.tf_w1]
        op_str = "dt,tp->dp"
        self.custom_einsum_f(op_str, my_args, tf_args, my_wrt=my_args, tf_wrt=tf_args)

    def test_threevars_f(self):
        my_args = [self.my_w0, self.my_w1, self.my_w2]
        tf_args = [self.tf_w0, self.tf_w1, self.tf_w2]
        op_str = "dt,tp,pr->dtp"
        self.custom_einsum_f(op_str, my_args, tf_args, my_wrt=my_args, tf_wrt=tf_args)

        # def test_summation(self):
        #     my_args = [self.my_w0, self.my_w1]
        #     tf_args = [self.tf_w0, self.tf_w1]
        #
        #     op_str = "dt,tp->d"
        #     with self.subTest(op_str):
        #         self.custom_einsum_f(op_str, my_args, tf_args)
        #
        #     op_str = "dt,tp->p"
        #     with self.subTest(op_str):
        #         self.custom_einsum_f(op_str, my_args, tf_args)
        #
        #     op_str = "dt,tp->"
        #     with self.subTest(op_str):
        #         self.custom_einsum_f(op_str, my_args, tf_args)


        # def test_threevars_df_third(self):
        #     tf_graph = tf.einsum("dt,tp,pr->dtp", self.tf_w0, self.tf_w1, self.tf_w2)
        #     my_graph = EinSum("dt,tp,pr->dtp", self.my_w0, self.my_w1, self.my_w2)
        #     with tf.Session():
        #         tf_grads = tf.gradients(tf_graph, self.tf_w2)[0].eval()
        #     print(tf_grads)
        #
        #     my_graph.compute_derivatives(self.my_input_dict)
        #     my_grads = my_graph.accumulate_all_gradients(wrt="w2_val")
        #
        #     print(my_grads)
        #     np.testing.assert_allclose(my_grads, tf_grads)
