# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function
from __future__ import absolute_import
from __future__ import division
import os
from termcolor import colored
from tensorflow.python import pywrap_tensorflow
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


def initialize_uninitialized(sess):
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

    print([str(i.name) for i in not_initialized_vars])  # only for testing
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))


def print_dashboard(type, step, batch_size, vocab_size,
                    running_avg_loss, eval_loss,
                    total_training_time, current_speed,
                    coverage_loss="not set"):
    print(
        "\nDashboard for %s updated %s, finished steps:\t%s\n"
        "\tBatch size:\t%s\n"
        "\tVocabulary size:\t%s\n"
        "\tArticles trained:\t%s\n"
        "\tTotal training time approxiately:\t%.4f hours\n"
        "\tCurrent speed:\t%.4f seconds/article\n"
        "\tTraining loss:\t%.4f; eval loss \t%.4f"
        "\tand coverage loss:\t%s\n" % (
            type,
            datetime.datetime.now().strftime("on %m-%d at %H:%M"),
            step,
            batch_size,
            vocab_size,
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


def add_encoder(encoder_inputs, seq_len, hidden_dim,
                rand_unif_init=None, state_is_tuple=True):
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
    return encoder_outputs, fw_st, bw_st


def reduce_states(fw_st, bw_st, hidden_dim, activation_fn=tf.tanh, trunc_norm_init_std=1e-4):
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
        return tf.contrib.rnn.LSTMStateTuple(new_c, new_h)  # Return new cell and state


def selective_fn(encoder_outputs, dec_in_state):
    enc_outputs = tf.transpose(encoder_outputs, perm=[1, 0, 2])
    dynamic_enc_steps = tf.shape(enc_outputs)[0]
    output_dim = encoder_outputs.get_shape()[-1]
    sele_ar = tf.TensorArray(dtype=tf.float32, size=dynamic_enc_steps)

    with tf.variable_scope('selective'):

        def cond(_e, i, _m):
            return i < dynamic_enc_steps

        def mask_fn(inputs, i, sele_ar):
            sGate = tf.sigmoid(
                linear(inputs[i], output_dim, True, scope="w") +
                linear([dec_in_state.h, dec_in_state.c], output_dim, True, scope="u"))
            sele_ar = sele_ar.write(i, inputs[i] * sGate)
            if i == tf.constant(0, dtype=tf.int32):
                tf.get_variable_scope().reuse_variables()
            return inputs, i+1, sele_ar

        _, _, sele_ar = tf.while_loop(
            cond, mask_fn, (enc_outputs, tf.constant(0, dtype=tf.int32), sele_ar))
        new_enc_outputs = tf.transpose(tf.squeeze(sele_ar.stack()), perm=[1, 0, 2])
    return new_enc_outputs


def sattolo_cycle(items):
    i = len(items)
    while i > 1:
        i = i - 1
        j = randrange(i)  # 0 <= j <= i-1
        items[j], items[i] = items[i], items[j]
