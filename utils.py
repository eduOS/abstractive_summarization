# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function
from __future__ import absolute_import
from __future__ import division
import os
from termcolor import colored
import datetime
import tensorflow as tf
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


def load_ckpt(saver, sess, dire, force=False, lastest_filename="checkpoint"):
    """Load checkpoint from the train directory and restore it to saver and sess,
    waiting 10 secs in the case of failure. Also returns checkpoint name."""
    while True:
        try:
            ckpt_state = tf.train.get_checkpoint_state(dire, lastest_filename)
            print('Loading checkpoint' + colored(' %s', 'yellow') % ckpt_state.model_checkpoint_path)
            saver.restore(sess, ckpt_state.model_checkpoint_path)
            return ckpt_state.model_checkpoint_path
        except Exception as ex:
            print(ex)
            print(colored("Failed to load checkpoint from %s" % dire, 'red'))
            if not force:
                return
            else:
                print("Failed to load checkpoint from %s Sleeping %s munites to waite." % (dire, 10))
                time.sleep(10)


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


def linear3d(args, output_size, bias, bias_start=0.0, scope=None):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

    Args:
      args: a 3D or 4D Tensor or a list of 3D or 4D, batch x n, Tensors.
      the second dimension of the 4D should be 1
      output_size: int, second dimension of W[i].
      bias: boolean, whether to add a bias term or not.
      bias_start: starting value to initialize the bias; 0 by default.
      scope: VariableScope for the created subgraph; defaults to "Linear".

    Returns:
      A 3D or 4D Tensor with shape [batch x output_size x emb_dim] equal to
      sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

    Raises:
      ValueError: if some of the arguments has unspecified or wrong shape.
    """
    if args is None or (isinstance(args, (list, tuple)) and not args):
        raise ValueError("`args` must be specified")
    if not isinstance(args, (list, tuple)):
        args = [args]
    flag_4d = len(args[0].get_shape().as_list()) == 4
    if flag_4d and args[0].get_shape()[1].value == 1:
        args = [tf.squeeze(arg, 1) for arg in args]
    elif flag_4d:
        raise ValueError("The second dimension of 4D should be 1.")

    emb_dim = args[0].get_shape().as_list()[-1]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    for n, shape in enumerate(shapes):
        if shape[2] != emb_dim:
            raise ValueError("The last dimension should be the same: %s(the first) but %s(the %sth)"
                             % (str(emb_dim), str(shape[2]), n))
        if len(shape) != 3:
            raise ValueError("Linear is expecting 3D arguments: %s" % str(shapes))
        if not shape[1]:
            raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
        else:
            total_arg_size += shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [output_size, total_arg_size])
        if len(args) > 1:
            args = tf.concat(axis=1, values=args)

        args = tf.unstack(args, axis=0)
        res = [tf.matmul(matrix, arg) for arg in args]
        res = tf.stack(args, axis=0)
        if not bias:
            return res
        bias_term = tf.get_variable("Bias", [output_size, emb_dim], initializer=tf.constant_initializer(bias_start))

    if flag_4d:
        output = tf.expand_dims(res + bias_term, 1)
    else:
        output = res + bias_term

    return output
