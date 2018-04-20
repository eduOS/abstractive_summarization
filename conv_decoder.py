# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
from utils import conv_block
from utils import _transpose_batch_time


def conv_decoder(decoder_inputs, enc_output, is_training,
                 embedding_dropout_keep_prob=0.9):

    # Apply dropout to embeddings
    inputs = tf.contrib.layers.dropout(
        inputs=decoder_inputs,
        keep_prob=embedding_dropout_keep_prob,
        is_training=is_training)

    next_layer = conv_block(enc_output, inputs, True)
    logits = _transpose_batch_time(next_layer)
    sample_ids = tf.cast(tf.argmax(logits, axis=-1), tf.int32)

    return logits, sample_ids
