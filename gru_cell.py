import tensorflow as tf
import numpy
# import random
# import time

from tensorflow.python.ops.rnn_cell import RNNCell
from tensorflow.python.ops import math_ops
# from tensorflow.python.ops import variable_scope
# from tensorflow.python.framework import ops
# from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops

from share_function import ortho_weight
from share_function import norm_weight


class GRULayer(RNNCell):
    """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

    def __init__(
        self,
        num_units,
        scope,
        input_size=None,
        activation=math_ops.tanh,
        init_device="/cpu:0",
        prefix='gru_layer',
        precision='float32',
     reuse_var=False):
        if input_size is not None:
            self._num_units = num_units
            self._activation = activation
            self._input_size = input_size
            self._scope = scope
            self._precision = precision
            if not reuse_var:
                with vs.variable_scope(self._scope or "gru_layer"):
                    with tf.device(init_device):
                        embDim = self._input_size
                        dim = self._num_units
                        W = numpy.concatenate(
                            [norm_weight(embDim, dim), norm_weight(embDim, dim)
                             ], axis=1)
                        W = tf.get_variable('W', initializer=tf.constant(W))
                        b = numpy.zeros((2 * dim,)).astype(self._precision)
                        b = tf.get_variable('b', initializer=tf.constant(b))

                        U = numpy.concatenate(
                            [ortho_weight(dim), ortho_weight(dim)], axis=1)
                        U = tf.get_variable('U', initializer=tf.constant(U))

                        Wx = norm_weight(embDim, dim)
                        Wx = tf.get_variable('Wx', initializer=tf.constant(Wx))

                        bx = numpy.zeros((dim,)).astype(self._precision)
                        bx = tf.get_variable('bx', initializer=tf.constant(bx))

                        Ux = ortho_weight(dim)
                        Ux = tf.get_variable('Ux', initializer=tf.constant(Ux))

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state=None):
        embs = inputs[0]

        # print(len(inputs))
        # def f1():
        #     return None
        # def f2():
        #     return inputs[1]

        # print('the shape of inputs ', inputs.get_shape())
        # condi = tf.less(tf.shape(inputs)[0], 2)

        # print('error found')

        # mask_slice = tf.cond(tf.less(tf.shape(inputs)[0], 2), f1, f2)

        if len(inputs) == 1:
            mask_slice = None
        else:
            mask_slice = inputs[1]

        """Gated recurrent unit (GRU) with nunits cells."""
        tf.get_variable_scope().reuse_variables()
        W = tf.get_variable('W', dtype=self._precision)
        b = tf.get_variable('b', dtype=self._precision)
        U = tf.get_variable('U', dtype=self._precision)
        Wx = tf.get_variable('Wx', dtype=self._precision)
        bx = tf.get_variable('bx', dtype=self._precision)
        Ux = tf.get_variable('Ux', dtype=self._precision)

        # graph build

        emb2hidden = math_ops.matmul(embs, Wx) + bx
        emb2gates = math_ops.matmul(embs, W) + b

        nsamples = tf.shape(embs)[0]
        if state is None:
            state = tf.zeros([nsamples, self._num_units], dtype=self._precision)

        if mask_slice is None:
            mask_slice = tf.ones([nsamples, self._num_units])  # for decoding

        # gates input for first gru layer
        preAct = math_ops.matmul(state, U)
        preAct += emb2gates
        preAct = math_ops.sigmoid(preAct)
        r, u = array_ops.split(1, 2, preAct)

        # hidden input for first gru layer
        preActx = math_ops.matmul(state, Ux)
        preActx *= r
        preActx += emb2hidden

        h = math_ops.tanh(preActx)

        h = u * state + (1. - u) * h
        h = mask_slice * h + (1. - mask_slice) * state

        new_h = h

        return new_h, new_h


class GRUCondLayer(RNNCell):

    def __init__(
        self,
        num_units,
        context,
        scope,
        context_mask=None,
        input_size=None,
        activation=math_ops.tanh,
        init_device="/cpu:0",
        prefix='gru_cond_layer',
        precision='float32',
     reuse_var=False):
        # logging.warn("%s: The input_size parameter is deprecated.", self)
        self._num_units = num_units
        self._activation = activation
        self._input_size = input_size
        self.context = context
        self.alphas = []
        self.ctxs = []
        self.context_mask = context_mask
        self.scope = scope
        self._precision = precision

        if not reuse_var:
            with vs.variable_scope(self.scope or "gru_cond_layer"):
                with tf.device(init_device):
                    embDim = self._input_size
                    dim = self._num_units

                    W = numpy.concatenate(
                        [norm_weight(embDim, dim),
                         norm_weight(embDim, dim)],
                        axis=1)
                    W = tf.get_variable(
                        'W', initializer=tf.constant(W), trainable=True)
                    b = numpy.zeros((2 * dim,)).astype(self._precision)
                    b = tf.get_variable(
                        'b', initializer=tf.constant(b), trainable=True)

                    U = numpy.concatenate(
                        [ortho_weight(dim), ortho_weight(dim)], axis=1)
                    U = tf.get_variable(
                        'U', initializer=tf.constant(U), trainable=True)

                    Wx = norm_weight(embDim, dim)
                    Wx = tf.get_variable(
                        'Wx', initializer=tf.constant(Wx), trainable=True)

                    Ux = ortho_weight(dim)
                    Ux = tf.get_variable(
                        'Ux', initializer=tf.constant(Ux), trainable=True)

                    bx = numpy.zeros((dim,)).astype(self._precision)
                    bx = tf.get_variable(
                        'bx', initializer=tf.constant(bx), trainable=True)

                    U_nl = numpy.concatenate(
                        [ortho_weight(dim), ortho_weight(dim)], axis=1)
                    U_nl = tf.get_variable(
                        'U_nl', initializer=tf.constant(U_nl), trainable=True)

                    b_nl = numpy.zeros((2 * dim,)).astype(self._precision)
                    b_nl = tf.get_variable(
                        'b_nl', initializer=tf.constant(b_nl), trainable=True)

                    Ux_nl = ortho_weight(dim)
                    Ux_nl = tf.get_variable(
                        'Ux_nl', initializer=tf.constant(Ux_nl), trainable=True)

                    bx_nl = numpy.zeros((dim,)).astype(self._precision)
                    bx_nl = tf.get_variable(
                        'bx_nl', initializer=tf.constant(bx_nl), trainable=True)

                    Wc = norm_weight(dim * 2, dim * 2)
                    Wc = tf.get_variable(
                        'Wc', initializer=tf.constant(Wc), trainable=True)

                    Wcx = norm_weight(dim * 2, dim)
                    Wcx = tf.get_variable(
                        'Wcx', initializer=tf.constant(Wcx), trainable=True)

                    W_comb_att = norm_weight(dim, dim * 2)
                    W_comb_att = tf.get_variable(
                        'W_comb_att', initializer=tf.constant(W_comb_att),
                        trainable=True)

                    Wc_att = norm_weight(dim * 2)
                    Wc_att = tf.get_variable(
                        'Wc_att', initializer=tf.constant(Wc_att),
                        trainable=True)

                    b_att = numpy.zeros((dim * 2,)).astype(self._precision)
                    b_att = tf.get_variable(
                        'b_att', initializer=tf.constant(b_att), trainable=True)

                    U_att = norm_weight(dim * 2, 1)
                    U_att = tf.get_variable(
                        'U_att', initializer=tf.constant(U_att), trainable=True)

                    c_tt = numpy.zeros((1,)).astype(self._precision)
                    c_tt = tf.get_variable(
                        'c_tt', initializer=tf.constant(c_tt), trainable=True)

    def set_pctx_(self):
        with vs.variable_scope(self.scope):
            tf.get_variable_scope().reuse_variables()
            context = self.context
            Wc_att = tf.get_variable('Wc_att', dtype=self._precision)
            b_att = tf.get_variable('b_att', dtype=self._precision)
            context_shape = tf.shape(context)
            context_2d = tf.reshape(context, [-1, context_shape[2]])
            pctx_ = math_ops.matmul(context_2d, Wc_att) + b_att
            pctx_ = tf.reshape(
                pctx_,
                [context_shape[0],
                 context_shape[1],
                 2 * self._num_units])
            self.pctx_ = pctx_

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units * 3

    def __call__(self, inputs, state):
        embs = inputs[0]
        if len(inputs) == 2:
            mask_slice = inputs[1]
        else:
            mask_slice = None

        context = self.context
        context_mask = self.context_mask
        pctx_ = self.pctx_
        """Gated recurrent unit (GRU) with nunits cells."""
        tf.get_variable_scope().reuse_variables()
        W = tf.get_variable('W', dtype=self._precision)
        b = tf.get_variable('b', dtype=self._precision)
        U = tf.get_variable('U', dtype=self._precision)
        Wx = tf.get_variable('Wx', dtype=self._precision)
        Ux = tf.get_variable('Ux', dtype=self._precision)
        bx = tf.get_variable('bx', dtype=self._precision)
        U_nl = tf.get_variable('U_nl', dtype=self._precision)
        b_nl = tf.get_variable('b_nl', dtype=self._precision)
        Ux_nl = tf.get_variable('Ux_nl', dtype=self._precision)
        bx_nl = tf.get_variable('bx_nl', dtype=self._precision)
        Wc = tf.get_variable('Wc', dtype=self._precision)
        Wcx = tf.get_variable('Wcx', dtype=self._precision)
        W_comb_att = tf.get_variable('W_comb_att', dtype=self._precision)
        # Wc_att = tf.get_variable('Wc_att', dtype=self._precision)
        # b_att = tf.get_variable('b_att', dtype=self._precision)
        U_att = tf.get_variable('U_att', dtype=self._precision)
        c_tt = tf.get_variable('c_tt', dtype=self._precision)

        # graph build
        emb2hidden = math_ops.matmul(embs, Wx) + bx
        emb2gates = math_ops.matmul(embs, W) + b

        nlocation = tf.shape(context)[0]
        nsamples = tf.shape(context)[1]
        if state is None:
            raise ValueError("init state must be provided.")

        if mask_slice is None:
            mask_slice = tf.ones([nsamples, self._num_units])  # for decoding

        # gates input for first gru layer
        preAct1 = math_ops.matmul(state, U)
        preAct1 += emb2gates
        preAct1 = math_ops.sigmoid(preAct1)
        r1, u1 = array_ops.split(1, 2, preAct1)

        # hidden input for first gru layer
        preActx1 = math_ops.matmul(state, Ux)
        preActx1 *= r1
        preActx1 += emb2hidden

        h1 = math_ops.tanh(preActx1)

        h1 = u1 * state + (1. - u1) * h1
        h1 = mask_slice * h1 + (1. - mask_slice) * state

        # attention
        pstate_ = math_ops.matmul(h1, W_comb_att)
        pctx__ = pctx_ + pstate_[None, :, :]
        pctx__ = math_ops.tanh(pctx__)

        pctx_2d = tf.reshape(pctx__, [-1, tf.shape(pctx__)[2]])
        alpha = math_ops.matmul(pctx_2d, U_att) + c_tt
        # alpha = math_ops.matmul(pctx__, U_att) + c_tt
        alpha = tf.reshape(alpha, [nlocation, nsamples])
        alpha = math_ops.exp(alpha)
        if context_mask is not None:
            alpha = alpha * context_mask

        alpha = alpha / tf.reduce_sum(alpha, 0, keep_dims=True)
        ctx_ = tf.reduce_sum(context * alpha[:, :, None], 0)

        preAct2 = math_ops.matmul(h1, U_nl) + b_nl
        preAct2 += math_ops.matmul(ctx_, Wc)
        preAct2 = math_ops.sigmoid(preAct2)

        r2, u2 = array_ops.split(1, 2, preAct2)

        preActx2 = math_ops.matmul(h1, Ux_nl)+bx_nl
        preActx2 *= r2
        preActx2 += math_ops.matmul(ctx_, Wcx)

        h2 = math_ops.tanh(preActx2)

        h2 = u2 * h1 + (1. - u2) * h2
        h2 = mask_slice * h2 + (1. - mask_slice) * h1

        output = tf.concat(1, [h2, ctx_])
        # if FLAGS.DebugMode:
        #        tensorPrint={}
        #        tensorPrint['emb2hidden'] = emb2hidden
        #        tensorPrint['emb2gates'] = emb2gates
        #        tensorPrint['preAct1'] = preAct1
        #        tensorPrint['preActx1'] = preActx1
        #        tensorPrint['h1'] = h1
        #        tensorPrint['alpha'] = alpha
        #        tensorPrint['preAct2'] = preAct2
        #        tensorPrint['preActx2'] = preActx2
        #        self.tensorPrint = tensorPrint

        return output, h2
