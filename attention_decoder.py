# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""This file defines the decoder"""
from __future__ import unicode_literals, print_function
from __future__ import absolute_import
from __future__ import division


import tensorflow as tf
# from gen_utils import get_local_global_features
from utils import linear_mapping_weightnorm
# from utils import global_selective_fn
from utils import conv_decoder_stack


# Note: this function is based on tf.contrib.legacy_seq2seq_attention_decoder,
# which is now outdated.
# In the future, it would make more sense to write variants on the
# attention mechanism using the new seq2seq library for tensorflow 1.0:
# https://www.tensorflow.org/api_guides/python/contrib.seq2seq#Attention

def conv_attention_decoder(emb_dec_inputs, enc_padding_mask, attention_keys, attention_values,
                           vocab_size, is_training, cnn_layers=4, nout_embed=256,
                           nhids_list=[256, 256, 256, 256], kwidths_list=[3, 3, 3, 3],
                           embedding_dropout_keep_prob=0.9, nhid_dropout_keep_prob=0.9, out_dropout_keep_prob=0.9):

    input_shape = emb_dec_inputs.get_shape().as_list()    # static shape. may has None
    # Apply dropout to embeddings
    inputs = tf.contrib.layers.dropout(
        inputs=emb_dec_inputs,
        keep_prob=embedding_dropout_keep_prob,
        is_training=is_training)

    with tf.variable_scope("decoder_cnn"):
        next_layer = inputs
        if cnn_layers > 0:

            # mapping emb dim to hid dim
            next_layer = linear_mapping_weightnorm(
                next_layer, nhids_list[0], dropout=embedding_dropout_keep_prob,
                var_scope_name="linear_mapping_before_cnn")

            next_layer = conv_decoder_stack(
                inputs, attention_keys, attention_values, next_layer, enc_padding_mask,
                nhids_list, kwidths_list, {'src': embedding_dropout_keep_prob, 'hid': nhid_dropout_keep_prob}, is_training=is_training)

    with tf.variable_scope("softmax"):
        if is_training:
            next_layer = linear_mapping_weightnorm(next_layer, nout_embed, var_scope_name="linear_mapping_after_cnn")
        else:
            # for refer it takes only the last one
            next_layer = linear_mapping_weightnorm(next_layer[:, -1:, :], nout_embed, var_scope_name="linear_mapping_after_cnn")
        outputs = tf.contrib.layers.dropout(
            inputs=next_layer,
            keep_prob=out_dropout_keep_prob,
            is_training=is_training)

    logits = linear_mapping_weightnorm(outputs, vocab_size, dropout=out_dropout_keep_prob, var_scope_name="logits_before_softmax")
    # reshape for the length to unstack
    if is_training:
        logits = tf.reshape(logits, [-1, input_shape[1], vocab_size])
    else:
        logits = tf.reshape(logits, [-1, vocab_size])

    return logits
