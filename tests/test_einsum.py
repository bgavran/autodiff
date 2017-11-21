from unittest import TestCase

import numpy as np
import tensorflow as tf

import autodiff as ad
from tests import utils


class TestEinSum(TestCase):
    def setUp(self):
        np.random.seed(1337)
        self.w0_val = np.random.randn(2, 3)
        self.w1_val = np.random.randn(3, 5)
        self.w2_val = np.random.randn(5, 5)
        self.w3_val = np.random.randn(2, 3, 5, 7)
        self.w4_val = np.random.randn(5)

        self.tf_w0 = tf.constant(self.w0_val)
        self.tf_w1 = tf.constant(self.w1_val)
        self.tf_w2 = tf.constant(self.w2_val)
        self.tf_w3 = tf.constant(self.w3_val)

        self.my_w0 = ad.Variable(self.w0_val, name="w0_val")
        self.my_w1 = ad.Variable(self.w1_val, name="w1_val")
        self.my_w2 = ad.Variable(self.w2_val, name="w2_val")
        self.my_w3 = ad.Variable(self.w3_val, name="w3_val")
        self.my_w4 = ad.Variable(self.w4_val, name="w4_val")

    def test_reducesumkeepdims1(self):
        axes = [0, 2]
        my_graph = ad.ReduceSumKeepDims(self.my_w3, axes=axes)
        wrt_vars = [self.my_w3]
        utils.custom_test(self, my_graph, wrt_vars)

    def test_ellipsis1(self):
        my_args = [self.my_w1, self.my_w4]
        op_str = "...k,k->...k"
        self.custom_einsum_f(op_str, my_args, my_wrt=my_args)

    def test_onearg_identity(self):
        my_args = [self.my_w3]
        tf_args = [self.tf_w3]
        op_str = "ijkl->ijkl"
        self.custom_einsum_f(op_str, my_args, tf_args=tf_args, my_wrt=my_args, tf_wrt=tf_args)

    def test_onearg_sum1(self):
        my_args = [self.my_w3]
        tf_args = [self.tf_w3]
        op_str = "ijkl->ijk"
        self.custom_einsum_f(op_str, my_args, tf_args=tf_args, my_wrt=my_args, tf_wrt=tf_args)

    def test_onearg_sum2(self):
        my_args = [self.my_w3]
        tf_args = [self.tf_w3]
        op_str = "ijkl->ij"
        self.custom_einsum_f(op_str, my_args, tf_args=tf_args, my_wrt=my_args, tf_wrt=tf_args)

    def test_onearg_sum3(self):
        my_args = [self.my_w3]
        tf_args = [self.tf_w3]
        op_str = "ijkl->i"
        self.custom_einsum_f(op_str, my_args, tf_args=tf_args, my_wrt=my_args, tf_wrt=tf_args)

    def test_onearg_sum4(self):
        my_args = [self.my_w3]
        tf_args = [self.tf_w3]
        op_str = "ijkl->i"
        self.custom_einsum_f(op_str, my_args, tf_args=tf_args, my_wrt=my_args, tf_wrt=tf_args)

    def test_onearg_sum5(self):
        my_args = [self.my_w3]
        tf_args = [self.tf_w3]
        op_str = "ijkl->"
        self.custom_einsum_f(op_str, my_args, tf_args=tf_args, my_wrt=my_args, tf_wrt=tf_args)

    def test_twovars_f(self):
        my_args = [self.my_w0, self.my_w1]
        tf_args = [self.tf_w0, self.tf_w1]
        op_str = "dt,tp->dp"
        self.custom_einsum_f(op_str, my_args, tf_args=tf_args, my_wrt=my_args, tf_wrt=tf_args)

    def test_threevars_f(self):
        my_args = [self.my_w0, self.my_w1, self.my_w2]
        tf_args = [self.tf_w0, self.tf_w1, self.tf_w2]
        op_str = "dt,tp,pr->dtp"
        self.custom_einsum_f(op_str, my_args, tf_args=tf_args, my_wrt=my_args, tf_wrt=tf_args)

    def test_summation1(self):
        my_args = [self.my_w0, self.my_w1]
        tf_args = [self.tf_w0, self.tf_w1]
        op_str = "dt,tp->d"
        self.custom_einsum_f(op_str, my_args, tf_args=tf_args, my_wrt=my_args, tf_wrt=tf_args)

    def test_summation2(self):
        my_args = [self.my_w0, self.my_w1]
        tf_args = [self.tf_w0, self.tf_w1]
        op_str = "dt,tp->p"
        self.custom_einsum_f(op_str, my_args, tf_args=tf_args, my_wrt=my_args, tf_wrt=tf_args)

    def test_summation3(self):
        my_args = [self.my_w0, self.my_w1]
        tf_args = [self.tf_w0, self.tf_w1]
        op_str = "dt,tp->"
        self.custom_einsum_f(op_str, my_args, tf_args=tf_args, my_wrt=my_args, tf_wrt=tf_args)

    def test_threevar_summation(self):
        my_args = [self.my_w0, self.my_w1, self.my_w2]
        tf_args = [self.tf_w0, self.tf_w1, self.tf_w2]
        op_str = "dt,tp,pr->dtp"
        self.custom_einsum_f(op_str, my_args, tf_args=tf_args, my_wrt=my_args, tf_wrt=tf_args)

    def custom_einsum_f(self, op_str, my_args, my_wrt, tf_args=None, tf_wrt=None):
        my_graph = ad.Einsum(op_str, *my_args)
        if tf_args and tf_wrt is not None:
            tf_graph = tf.einsum(op_str, *tf_args)
            utils.custom_test(self, my_graph, my_wrt, tf_graph, tf_wrt)
        else:
            utils.custom_test(self, my_graph, my_wrt)
