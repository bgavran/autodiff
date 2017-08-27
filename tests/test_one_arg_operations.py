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

        np.testing.assert_allclose(my_val, tf_val)

    def oneop_df_wrt_w0(self, var_op, tf_op):
        tf_graph = tf_op(self.tf_w0)
        graph = var_op(self.var_w0)
        with tf.Session():
            tf_grads = tf.gradients(tf_graph, self.tf_w0)[0].eval()

        grad_ops = Grad(graph, wrt=self.var_w0)
        my_grads = grad_ops.eval(self.input_dict)

        # TODO this might not be such a good idea?
        # Is there a way to dynamically get the shape of the gradient?
        # Here I'm just relying on broadcasting to work
        np.testing.assert_allclose(my_grads + self.w0, tf_grads + self.w0)

    def oneop_2df_wrt_w0(self, var_op, tf_op):
        tf_graph = tf_op(self.tf_w0)
        graph = var_op(self.var_w0)
        with tf.Session():
            tf_grads0 = tf.gradients(tf_graph, self.tf_w0)[0]
            tf_grads1 = tf.gradients(tf_grads0, self.tf_w0)[0]
            if tf_grads1 is not None:
                tf_grads1 = tf_grads1.eval()
            else:
                tf_grads1 = 0

        grad_ops0 = Grad(graph, wrt=self.var_w0)
        grad_ops1 = Grad(grad_ops0, wrt=self.var_w0)
        my_grads = grad_ops1.eval(self.input_dict)

        np.testing.assert_allclose(my_grads + self.w0, tf_grads1 + self.w0)

    def oneop(self, var_op, tf_op):
        with self.subTest("f"):
            self.oneop_f(var_op, tf_op)
        with self.subTest("df"):
            self.oneop_df_wrt_w0(var_op, tf_op)
        with self.subTest("2df"):
            self.oneop_2df_wrt_w0(var_op, tf_op)

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
