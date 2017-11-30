# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function
from __future__ import absolute_import
from __future__ import division
import os
import tensorflow as tf
import time


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


def load_ckpt(saver, sess, dire, force=False):
    """Load checkpoint from the train directory and restore it to saver and sess,
    waiting 10 secs in the case of failure. Also returns checkpoint name."""
    while True:
        try:
            ckpt_state = tf.train.get_checkpoint_state(dire)
            print('Loading checkpoint %s' % ckpt_state.model_checkpoint_path)
            saver.restore(sess, ckpt_state.model_checkpoint_path)
            return ckpt_state.model_checkpoint_path
        except:
            print("Failed to load checkpoint from %s." % dire)
            if not force:
                return
            else:
                print("Failed to load checkpoint from %s. Sleeping %s munites to waite." % (dire, 10))
                time.sleep(10)
