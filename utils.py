# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function
from __future__ import absolute_import
from __future__ import division
import os
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
            print('Loading checkpoint %s' % ckpt_state.model_checkpoint_path)
            saver.restore(sess, ckpt_state.model_checkpoint_path)
            return ckpt_state.model_checkpoint_path
        except Exception as ex:
            print(ex)
            print("Failed to load checkpoint from %s" % dire)
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
                break
            else:
                sample_padding_mask[i][j] = 1
    return sample_padding_mask
