# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function
from __future__ import absolute_import
from __future__ import division
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import gen_array_ops

from tensorflow.python.layers import base

import os
from termcolor import colored
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.contrib.rnn import LSTMStateTuple
import datetime
import tensorflow as tf
from random import randrange
import time
import numpy as np
import data


def ensure_exists(dire):
    if not os.path.exists(dire):
        os.makedirs(dire)
    return dire


def get_config():
    """Returns config for tf.session"""
    config = tf.ConfigProto(allow_soft_placement=True)
    # config.gpu_options.per_process_gpu_memory_fraction = 0.9
    config.gpu_options.allow_growth = True
    return config


def load_ckpt(saver, sess, dire, mode="train", force=False, lastest_filename="checkpoint"):
    """Load checkpoint from the train directory and restore it to saver and sess,
    waiting 10 secs in the case of failure. Also returns checkpoint name."""
    while True:
        try:
            first_ckpt_dir = dire
            ckpt_state = tf.train.get_checkpoint_state(first_ckpt_dir, lastest_filename)
            print('Loading checkpoint' + colored(' %s', 'yellow') % ckpt_state.model_checkpoint_path)
            saver.restore(sess, ckpt_state.model_checkpoint_path)
            return ckpt_state.model_checkpoint_path
        except Exception as ex:
            print(colored("Failed to load checkpoint from %s. " % first_ckpt_dir, 'red'))
            try:
                if mode == "train":
                    second_ckpt_dir = os.path.join(dire, "val")
                else:
                    second_ckpt_dir = os.path.split(dire)[0]
                ckpt_state = tf.train.get_checkpoint_state(second_ckpt_dir, lastest_filename)
                print('Loading checkpoint' + colored(' %s', 'yellow') % ckpt_state.model_checkpoint_path)
                saver.restore(sess, ckpt_state.model_checkpoint_path)
                return ckpt_state.model_checkpoint_path
            except Exception as ex:
                print(ex)
                if not force:
                    print(colored("Failed to load checkpoint from %s also. Training from scratch.." % (second_ckpt_dir), 'red'))
                    return None
                elif mode == "train":
                    print("Failed to load checkpoint from %s Sleeping %s munites to waite." % (second_ckpt_dir, 10))
                    time.sleep(10 * 60)
        time.sleep(60)


def initialize_uninitialized(sess):
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

    print([str(i.name) for i in not_initialized_vars])  # only for testing
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))


def print_dashboard(type, step, batch_size, enc_vocab_size, dec_vocab_size,
                    running_avg_loss, eval_loss,
                    total_training_time, current_speed, current_learning_rate,
                    coverage_loss="not set"):
    print(
        "\nDashboard for %s updated %s, finished steps:\t%s\n"
        "\tBatch size:\t%s, current learning rate:\t%s\n"
        "\tEncoder vocabulary size:\t%s\n"
        "\tDecoder vocabulary size:\t%s\n"
        "\tArticles trained:\t%s\n"
        "\tTotal training time approxiately:\t%.4f hours\n"
        "\tCurrent speed:\t%.4f seconds/article\n"
        "\tTraining loss:\t%.4f; eval loss \t%.4f"
        "\tand coverage loss:\t%s\n" % (
            type,
            datetime.datetime.now().strftime("on %m-%d at %H:%M"),
            step,
            batch_size,
            current_learning_rate,
            enc_vocab_size,
            dec_vocab_size,
            batch_size * step,
            total_training_time,
            current_speed,
            running_avg_loss, eval_loss,
            coverage_loss,
            )
    )


def pad_sample(best_samples, vocab, hps):
    sample_padding_mask = np.zeros((hps.batch_size, hps.max_dec_steps), dtype=np.float32)

    # Fill in the numpy arrays
    for i, sp in enumerate(best_samples):
        for j, p in enumerate(sp):
            if p == vocab.word2id(data.STOP_DECODING):
                sample_padding_mask[i][j] = 1
                break
            else:
                sample_padding_mask[i][j] = 1
    return sample_padding_mask


def linear(args, output_size, bias, bias_start=0.0, scope=None):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

    Args:
      args: a 2D Tensor or a list of 2D, batch x n, Tensors.
      output_size: int, second dimension of W[i].
      bias: boolean, whether to add a bias term or not.
      bias_start: starting value to initialize the bias; 0 by default.
      scope: VariableScope for the created subgraph; defaults to "Linear".

    Returns:
      A 2D Tensor with shape [batch x output_size] equal to
      sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

    Raises:
      ValueError: if some of the arguments has unspecified or wrong shape.
    """
    if args is None or (isinstance(args, (list, tuple)) and not args):
        raise ValueError("`args` must be specified")
    if not isinstance(args, (list, tuple)):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
        if not shape[1]:
            raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
        else:
            total_arg_size += shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [total_arg_size, output_size])
        if len(args) == 1:
            res = tf.matmul(args[0], matrix)
        else:
            res = tf.matmul(tf.concat(axis=1, values=args), matrix)
        if not bias:
            return res
        bias_term = tf.get_variable("Bias", [output_size], initializer=tf.constant_initializer(bias_start))
    return res + bias_term


def linear_mapping_weightnorm(inputs, out_dim, dropout=1.0, var_scope_name="linear_mapping"):
  with tf.variable_scope(var_scope_name):
    input_shape = inputs.get_shape().as_list()    # static shape. may has None
    input_shape_tensor = tf.shape(inputs)
    #  use weight normalization (Salimans & Kingma, 2016)  w = g* v/2-norm(v)
    V = tf.get_variable(
        'V', shape=[int(input_shape[-1]), out_dim], dtype=tf.float32,
        initializer=tf.random_normal_initializer(mean=0, stddev=tf.sqrt(dropout*1.0/int(input_shape[-1]))), trainable=True)
    V_norm = tf.norm(V.initialized_value(), axis=0)  # V shape is M*N,  V_norm shape is N
    g = tf.get_variable('g', dtype=tf.float32, initializer=V_norm, trainable=True)
    b = tf.get_variable('b', shape=[out_dim], dtype=tf.float32, initializer=tf.zeros_initializer(), trainable=True)   # weightnorm bias is init zero

    assert len(input_shape) == 3
    inputs = tf.reshape(inputs, [-1, input_shape[-1]])
    inputs = tf.matmul(inputs, V)
    if input_shape[-2] is not None:
        inputs = tf.reshape(inputs, [input_shape_tensor[0], input_shape[-2], out_dim])
    else:
        inputs = tf.reshape(inputs, [input_shape_tensor[0], -1, out_dim])
    # inputs = tf.matmul(inputs, V)    # x*v

    scaler = tf.div(g, tf.norm(V, axis=0))   # g/2-norm(v)
    inputs = tf.reshape(scaler, [1, out_dim])*inputs + tf.reshape(b, [1, out_dim])  # x*v g/2-norm(v) + b

    return inputs


def variable_names_from_dir(chpt_dir, name_filter=""):
    ckpt = tf.train.get_checkpoint_state(chpt_dir)
    if ckpt:
        reader = pywrap_tensorflow.NewCheckpointReader(ckpt.model_checkpoint_path)
        var_to_shape_map = reader.get_variable_to_shape_map()
        variable_names = [key for key in var_to_shape_map if name_filter in key]
    return variable_names


def red_assert(statement, message, color='red'):
    assert statement, colored(message, color)


def red_print(message, color='red'):
    print(colored(message, color))


def lstm_encoder(encoder_inputs, seq_len, hidden_dim, rand_unif_init=None,
                 state_is_tuple=True, trunc_norm_init_std=1e-4,
                 keep_prob=0.9, is_training=False
                 ):
    """Add a single-layer bidirectional LSTM encoder to the graph.

    Args:
        encoder_inputs: A tensor of shape [batch_size, <=max_enc_steps,
        emb_size].
        seq_len: Lengths of encoder_inputs (before padding). A tensor of shape
        [batch_size].

    Returns:
        encoder_outputs:
        A tensor of shape [batch_size, <=max_enc_steps, 2*hidden_dim]. It's
        2*hidden_dim because it's the concatenation of the forwards and
        backwards states.
        fw_state, bw_state:
        Each are LSTMStateTuples of shape
        ([batch_size,hidden_dim],[batch_size,hidden_dim])
    """
    encoder_outputs = tf.contrib.layers.dropout(encoder_inputs, keep_prob=keep_prob, is_training=is_training)

    with tf.variable_scope('encoder'):
        cell_fw = tf.contrib.rnn.LSTMCell(
            hidden_dim, initializer=rand_unif_init, state_is_tuple=state_is_tuple)
        cell_bw = tf.contrib.rnn.LSTMCell(
            hidden_dim, initializer=rand_unif_init, state_is_tuple=state_is_tuple)
        (encoder_outputs, (fw_st, bw_st)) = tf.nn.bidirectional_dynamic_rnn(
            cell_fw, cell_bw, encoder_inputs, dtype=tf.float32, sequence_length=seq_len)
        # the sequence length of the encoder_inputs varies depending on the
        # batch, which will make the second dimension of the
        # encoder_outputs different in different batches

        # concatenate the forwards and backwards states
        encoder_outputs = tf.concat(axis=2, values=encoder_outputs)
        # encoder_outputs: [batch_size * beam_size, max_time, output_size*2]
        # fw_st & bw_st: [batch_size * beam_size, num_hidden]

    dec_in_state = reduce_states(
        fw_st, bw_st, hidden_dim=hidden_dim,
        activation_fn=tf.tanh, trunc_norm_init_std=trunc_norm_init_std,
        keep_prob=keep_prob, is_training=is_training)
    encoder_outputs = tf.contrib.layers.dropout(encoder_outputs, keep_prob=keep_prob, is_training=is_training)
    return encoder_outputs, dec_in_state


def conv_encoder(inputs, seq_len, is_training,
                 keep_prob=0.9, cnn_layers=6,
                 nhids_list=[256, 256, 256, 256, 256, 256],
                 kwidths_list=[3, 3, 3, 3, 3, 3]):
    embed_size = inputs.get_shape().as_list()[-1]

    #  Apply dropout to embeddings
    inputs = tf.contrib.layers.dropout(
        inputs=inputs,
        keep_prob=keep_prob,
        is_training=is_training)

    with tf.variable_scope("encoder_cnn"):
        next_layer = inputs
        if cnn_layers > 0:
            # mapping emb dim to hid dim
            next_layer = linear_mapping_weightnorm(next_layer, nhids_list[0], dropout=keep_prob, var_scope_name="linear_mapping_before_cnn")
            attentions_keys, next_layer = conv_encoder_stack(next_layer, nhids_list, kwidths_list, {'src': keep_prob, 'hid': keep_prob}, is_training=is_training)
            next_layer = linear_mapping_weightnorm(next_layer, embed_size, var_scope_name="linear_mapping_after_cnn")
            #  The encoder stack will receive gradients *twice* for each attention pass: dot product and weighted sum.
            #  cnn = nn.GradMultiply(cnn, 1 / (2 * nattn))
        cnn_c_output = (next_layer + inputs) * tf.sqrt(0.5)

    final_state = tf.reduce_mean(cnn_c_output, 1)
    return attentions_keys, final_state


def conv_encoder_stack(inputs, nhids_list, kwidths_list, dropout_dict, is_training):
    next_layer = inputs
    attentions_keys = []
    for layer_idx in range(len(nhids_list)):
        nin = nhids_list[layer_idx] if layer_idx == 0 else nhids_list[layer_idx-1]
        nout = nhids_list[layer_idx]
        if nin != nout:
            # mapping for res add
            res_inputs = linear_mapping_weightnorm(next_layer, nout, dropout=dropout_dict['src'], var_scope_name="linear_mapping_cnn_" + str(layer_idx))
        else:
            res_inputs = next_layer
        # dropout before input to conv
        next_layer = tf.contrib.layers.dropout(
            inputs=next_layer,
            keep_prob=dropout_dict['hid'],
            is_training=is_training)

        next_layer = conv1d_weightnorm(inputs=next_layer, layer_idx=layer_idx, out_dim=nout*2, kernel_size=kwidths_list[layer_idx], padding="SAME", dropout=dropout_dict['hid'], var_scope_name="conv_layer_"+str(layer_idx))
        next_layer = gated_linear_units(next_layer)
        attentions_keys.append(next_layer)
        next_layer = (next_layer + res_inputs) * tf.sqrt(0.5)
    attentions_keys = tf.stack(attentions_keys)
    return attentions_keys, next_layer


def gated_linear_units(inputs):
    input_shape = inputs.get_shape().as_list()
    assert len(input_shape) == 3
    input_pass = inputs[:, :, 0:int(input_shape[2]/2)]
    input_gate = inputs[:, :, int(input_shape[2]/2):]
    input_gate = tf.sigmoid(input_gate)
    return tf.multiply(input_pass, input_gate)


def conv1d_weightnorm(inputs, layer_idx, out_dim, kernel_size, padding="SAME", dropout=1.0,  var_scope_name="conv_layer"):
    #  TODO: padding should take attention

    with tf.variable_scope("conv_layer_"+str(layer_idx)):
        in_dim = int(inputs.get_shape()[-1])
        V = tf.get_variable('V', shape=[kernel_size, in_dim, out_dim], dtype=tf.float32, initializer=tf.random_normal_initializer(mean=0, stddev=tf.sqrt(4.0*dropout/(kernel_size*in_dim))), trainable=True)
        V_norm = tf.norm(V.initialized_value(), axis=[0, 1])
        # V shape is M*N*k,  V_norm shape is k
        g = tf.get_variable('g', dtype=tf.float32, initializer=V_norm, trainable=True)
        b = tf.get_variable('b', shape=[out_dim], dtype=tf.float32, initializer=tf.zeros_initializer(), trainable=True)

        # use weight normalization (Salimans & Kingma, 2016)
        W = tf.reshape(g, [1, 1, out_dim])*tf.nn.l2_normalize(V, [0, 1])
        inputs = tf.nn.bias_add(tf.nn.conv1d(value=inputs, filters=W, stride=1, padding=padding), b)
        return inputs


def reduce_states(fw_st, bw_st, hidden_dim, activation_fn=tf.tanh, trunc_norm_init_std=1e-4, keep_prob=0.5, is_training=True):
    """Add to the graph a linear layer to reduce the encoder's final FW and
    BW state into a single initial state for the decoder. This is needed
    because the encoder is bidirectional but the decoder is not.

    Args:
        fw_st: LSTMStateTuple with hidden_dim units.
        bw_st: LSTMStateTuple with hidden_dim units.

    Returns:
        state: LSTMStateTuple with hidden_dim units.
    """
    trunc_norm_init = tf.truncated_normal_initializer(stddev=trunc_norm_init_std)
    alpha = 0.01

    with tf.variable_scope('reduce_final_st'):

        # Define weights and biases to reduce the cell and reduce the state
        w_reduce_c = tf.get_variable(
            'w_reduce_c', [hidden_dim * 2, hidden_dim],
            dtype=tf.float32, initializer=trunc_norm_init)
        w_reduce_h = tf.get_variable(
            'w_reduce_h', [hidden_dim * 2, hidden_dim],
            dtype=tf.float32, initializer=trunc_norm_init)
        bias_reduce_c = tf.get_variable(
            'bias_reduce_c', [hidden_dim],
            dtype=tf.float32, initializer=trunc_norm_init)
        bias_reduce_h = tf.get_variable(
            'bias_reduce_h', [hidden_dim],
            dtype=tf.float32, initializer=trunc_norm_init)

        # Apply linear layer
        # Concatenation of fw and bw cell
        old_c = tf.concat(axis=1, values=[fw_st.c, bw_st.c])
        # Concatenation of fw and bw state
        old_h = tf.concat(axis=1, values=[fw_st.h, bw_st.h])
        # [batch_size * beam_size, hidden_dim]
        _c = tf.matmul(old_c, w_reduce_c) + bias_reduce_c
        _h = tf.matmul(old_h, w_reduce_h) + bias_reduce_h
        new_c = tf.nn.relu(_c) - alpha * tf.nn.relu(-_c)
        new_h = tf.nn.relu(_h) - alpha * tf.nn.relu(-_h)
        # new_c = activation_fn(tf.matmul(old_c, w_reduce_c) + bias_reduce_c)  # Get new cell from old cell
        # new_h = activation_fn(tf.matmul(old_h, w_reduce_h) + bias_reduce_h)  # Get new state from old state
        new_c = tf.contrib.layers.dropout(new_c, keep_prob=keep_prob, is_training=is_training)
        new_h = tf.contrib.layers.dropout(new_h, keep_prob=keep_prob, is_training=is_training)
        return tf.contrib.rnn.LSTMStateTuple(new_c, new_h)  # Return new cell and state


def selective_fn(encoder_outputs, dec_in_state, scope="selective_fn"):
    with tf.variable_scope(scope):
        enc_outputs = tf.transpose(encoder_outputs, perm=[1, 0, 2])
        dynamic_enc_steps = tf.shape(enc_outputs)[0]
        batch_size = encoder_outputs.get_shape().as_list()[0]
        output_dim = encoder_outputs.get_shape()[-1].value
        sele_ar = tf.TensorArray(dtype=tf.float32, size=dynamic_enc_steps)
        dec_state = [dec_in_state.h, dec_in_state.c] if type(dec_in_state) == LSTMStateTuple else dec_in_state

        with tf.variable_scope('selective'):

            def cond(_e, i, _m):
                return i < dynamic_enc_steps

            def mask_fn(inputs, i, sele_ar):
                sGate = tf.sigmoid(
                    linear(inputs[i], output_dim, True, scope="w") +
                    linear(dec_state, output_dim, True, scope="u"))
                sele_ar = sele_ar.write(i, inputs[i] * sGate)
                if i == tf.constant(0, tf.int32):
                    tf.get_variable_scope().reuse_variables()
                return inputs, i+1, sele_ar

            _, _, sele_ar = tf.while_loop(
                cond, mask_fn, (enc_outputs, tf.constant(0, dtype=tf.int32), sele_ar))
            new_enc_outputs = tf.transpose(tf.squeeze(sele_ar.stack()), perm=[1, 0, 2])
            new_enc_outputs = tf.reshape(new_enc_outputs, [batch_size, -1, output_dim])
    return new_enc_outputs


def sattolo_cycle(items):
    i = len(items)
    while i > 1:
        i = i - 1
        j = randrange(i)  # 0 <= j <= i-1
        items[j], items[i] = items[i], items[j]


def maxout(inputs, num_units, axis=-1, name=None):
    """Adds a maxout op from https://arxiv.org/abs/1302.4389
    "Maxout Networks" Ian J. Goodfellow, David Warde-Farley, Mehdi Mirza, Aaron
    Courville,
    Yoshua Bengio
    Usually the operation is performed in the filter/channel dimension. This can
    also be
    used after fully-connected layers to reduce number of features.
    Arguments:
    inputs: Tensor input
    num_units: Specifies how many features will remain after maxout in the `axis`
        dimension
            (usually channel). This must be multiple of number of `axis`.
    axis: The dimension where max pooling will be performed. Default is the
    last dimension.
    name: Optional scope for name_scope.
    Returns:
        A `Tensor` representing the results of the pooling operation.
    Raises:
        ValueError: if num_units is not multiple of number of features.
    """
    return MaxOut(num_units=num_units, axis=axis, name=name)(inputs)


class MaxOut(base.Layer):
    """Adds a maxout op from https://arxiv.org/abs/1302.4389
    "Maxout Networks" Ian J. Goodfellow, David Warde-Farley, Mehdi Mirza, Aaron
    Courville, Yoshua
    Bengio
    Usually the operation is performed in the filter/channel dimension. This can
    also be
    used after fully-connected layers to reduce number of features.
    Arguments:
        inputs: Tensor input
        num_units: Specifies how many features will remain after maxout in the
        `axis` dimension
            (usually channel).
        This must be multiple of number of `axis`.
        axis: The dimension where max pooling will be performed. Default is the
        last dimension.
        name: Optional scope for name_scope.
    Returns:
        A `Tensor` representing the results of the pooling operation.
    Raises:
        ValueError: if num_units is not multiple of number of features.
    """

    def __init__(self, num_units, axis=-1, name=None, **kwargs):
        super(MaxOut, self).__init__(name=name, trainable=False, **kwargs)
        self.axis = axis
        self.num_units = num_units

    def call(self, inputs):
        inputs = ops.convert_to_tensor(inputs)
        shape = inputs.get_shape().as_list()
        num_channels = shape[self.axis]
        if num_channels % self.num_units:
            raise ValueError('number of features({}) is not '
                             'a multiple of num_units({})'.format(
                                 num_channels, self.num_units))
        shape[self.axis] = -1
        shape += [num_channels // self.num_units]

    # Dealing with batches with arbitrary sizes
        for i in range(len(shape)):
            if shape[i] is None:
                shape[i] = gen_array_ops.shape(inputs)[i]
        outputs = math_ops.reduce_max(
            gen_array_ops.reshape(inputs, shape), -1, keep_dims=False)

        return outputs


def conv_decoder_stack(enc_inputs, dec_labels, attentions_keys, inputs, enc_padding_mask,
                       nhids_list, kwidths_list, dropout_dict, is_training):
    next_layer = inputs

    for layer_idx in range(len(nhids_list)):
        nin = nhids_list[layer_idx] if layer_idx == 0 else nhids_list[layer_idx-1]
        nout = nhids_list[layer_idx]
        if nin != nout:
            # mapping for res add
            res_inputs = linear_mapping_weightnorm(next_layer, nout, dropout=dropout_dict['hid'], var_scope_name="linear_mapping_cnn_" + str(layer_idx))
        else:
            res_inputs = next_layer
        # dropout before input to conv
        next_layer = tf.contrib.layers.dropout(
            inputs=next_layer,
            keep_prob=dropout_dict['hid'],
            is_training=is_training)
        # special process here, first padd then conv, because tf does not suport padding other than SAME and VALID
        next_layer = tf.pad(next_layer, [[0, 0], [kwidths_list[layer_idx]-1, kwidths_list[layer_idx]-1], [0, 0]], "CONSTANT")

        next_layer = conv1d_weightnorm(inputs=next_layer, layer_idx=layer_idx, out_dim=nout*2, kernel_size=kwidths_list[layer_idx], padding="VALID", dropout=dropout_dict['hid'], var_scope_name="conv_layer_"+str(layer_idx))
        layer_shape = next_layer.get_shape().as_list()
        assert len(layer_shape) == 3
        # to avoid using future information
        next_layer = next_layer[:, 0:-kwidths_list[layer_idx]+1, :]
        next_layer = gated_linear_units(next_layer)

        # add attention
        # decoder output -->linear mapping to embed, + target embed,  query decoder output a, softmax --> scores, scores*encoder_output_c-->output,  output--> linear mapping to nhid+  decoder_output -->
        att_out, attn_dist = multihead_attention(enc_inputs, dec_labels, attentions_keys, next_layer, layer_idx, enc_padding_mask)
        # att_out += linear_mapping_weightnorm(_att_out, _att_out.get_shape().as_list()[-1], "linear_mapping_att_out_"+str(layer_idx))
        next_layer = (next_layer + att_out) * tf.sqrt(0.5)

        # add res connections
        next_layer += (next_layer + res_inputs) * tf.sqrt(0.5)
        # why they are not accumulated in a list?

    # this remains a problem on how to combine the outs and scores in different
    # layers
    return next_layer, att_out, attn_dist


def linear_mapping_stupid(inputs, out_dim, in_dim=None, dropout=1.0, var_scope_name="linear_mapping"):
  with tf.variable_scope(var_scope_name):
    # print('name', tf.get_variable_scope().name)
    input_shape_tensor = tf.shape(inputs)   # dynamic shape, no None
    input_shape = inputs.get_shape().as_list()    # static shape. may has None
    # print('input_shape', input_shape)
    assert len(input_shape) == 3
    inputs = tf.reshape(inputs, [-1, input_shape_tensor[-1]])

    linear_mapping_w = tf.get_variable("linear_mapping_w", [input_shape[-1], out_dim], initializer=tf.random_normal_initializer(mean=0, stddev=tf.sqrt(dropout*1.0/input_shape[-1])))
    linear_mapping_b = tf.get_variable("linear_mapping_b", [out_dim], initializer=tf.zeros_initializer())

    output = tf.matmul(inputs, linear_mapping_w) + linear_mapping_b
    # print('xxxxx_params', input_shape, out_dim)
    # output = tf.reshape(output, [input_shape[0], -1, out_dim])
    output = tf.reshape(output, [input_shape_tensor[0], -1, out_dim])
  return output


def multihead_attention(enc_inputs, dec_labels, attentions_keys, decoder_hidden, layer_idx, enc_padding_mask):
    """
    this is the so called dot product attention
    TODO: the tf.sqrt(0.5) should be replaced to make the attention scaled dot product attention
    enc_padding_mask: M*N2

    query:
    key: attentions keys
    value:
    """

    att_outs = []
    att_scores = []
    attentions_keys = tf.unstack(attentions_keys)
    enc_padding_mask = tf.tile(tf.expand_dims(enc_padding_mask, axis=1),
                               [1, decoder_hidden.get_shape().as_list()[1], 1])

    with tf.variable_scope("attention_layer_" + str(layer_idx)):
        embed_size = dec_labels.get_shape().as_list()[-1]
        for k, attention_keys in enumerate(attentions_keys):
            # k
            dec_hidden_proj = linear_mapping_weightnorm(decoder_hidden, embed_size, var_scope_name="linear_mapping_att_query" + str(k))
            # M*N1*k1 --> M*N1*k
            att_query = (dec_hidden_proj + dec_labels) * tf.sqrt(0.5)
            attention_key_proj = linear_mapping_weightnorm(attention_keys, embed_size, var_scope_name="linear_mapping_att_key" + str(k))
            attention_value_proj = linear_mapping_weightnorm(enc_inputs, embed_size, var_scope_name="linear_mapping_att_value" + str(k))

            att_score = tf.matmul(att_query, attention_key_proj, transpose_b=True)
            # M*N1*K  ** M*N2*K  --> M*N1*N2
            att_score *= enc_padding_mask
            att_scores.append(att_score)
            att_score = tf.nn.softmax(att_score)

            length = tf.cast(tf.shape(attention_value_proj), tf.float32)
            att_out = tf.matmul(att_score, attention_value_proj) * length[1] * tf.sqrt(1.0/length[1])
            # M*N1*N2  ** M*N2*K   --> M*N1*k
            att_outs.append(att_out)

        att_out = tf.concat(axis=-1, values=att_outs)
        att_out = linear_mapping_weightnorm(att_out, decoder_hidden.get_shape().as_list()[-1], var_scope_name="linear_mapping_att_out")
        _att_scores = tf.stack(axis=-1, values=att_scores)
        SV = tf.get_variable(
            'SV', shape=[int(_att_scores.get_shape().as_list()[-1]), 1], dtype=tf.float32, trainable=True)
        _att_scores = tf.reshape(_att_scores, [-1, _att_scores.get_shape().as_list()[-1]])
        att_score = tf.matmul(_att_scores, SV)
        att_score = tf.reshape(
            att_score,
            [tf.shape(att_scores[0])[0], att_scores[0].get_shape().as_list()[1], tf.shape(att_scores[0])[-1]])
        att_score = tf.nn.softmax(att_score)
    return att_out, att_score


def transpose_batch_time(x):
    """
    Transpose the batch and time dimensions of a Tensor.
    Retains as much of the static shape information as possible.
    Args:
        x: A tensor of rank 2 or higher.
    Returns:
        x transposed along the first two dimensions.
    Raises:
        ValueError: if `x` is rank 1 or lower.
    """
    x_static_shape = x.get_shape()
    if x_static_shape.ndims is not None and x_static_shape.ndims < 2:
        raise ValueError(
            "Expected input tensor %s to have rank at least 2, but saw shape: %s" %
            (x, x_static_shape))
    x_rank = array_ops.rank(x)
    x_t = array_ops.transpose(
        x, array_ops.concat(([1, 0], math_ops.range(2, x_rank)), axis=0))
    x_t.set_shape(
        tensor_shape.TensorShape(
            [x_static_shape[1].value, x_static_shape[0].value]).concatenate(x_static_shape[2:]))
    return x_t


def label_sentence_num(sentence, stop_ids):
    if not isinstance(sentence, list):
        sentence = sentence.split()
    _label = [0] * len(sentence)
    count = 0
    for i, c in enumerate(sentence):
        _label[i] = str(count)
        if c in stop_ids:
            count += 1
    return _label
