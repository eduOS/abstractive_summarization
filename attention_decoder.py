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
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops
# from gen_utils import get_local_global_features
from utils import linear_mapping_weightnorm
# from utils import global_selective_fn
from utils import conv_decoder_stack
from utils import linear
from utils import maxout


# Note: this function is based on tf.contrib.legacy_seq2seq_attention_decoder,
# which is now outdated.
# In the future, it would make more sense to write variants on the
# attention mechanism using the new seq2seq library for tensorflow 1.0:
# https://www.tensorflow.org/api_guides/python/contrib.seq2seq#Attention

def lstm_attention_decoder(decoder_inputs, enc_sent_label, enc_padding_mask, attention_keys,
                           initial_state, cell, initial_state_attention=False, use_coverage=False,
                           prev_coverage=None, local_attention_layers=3):
    assert type(decoder_inputs) == list, "decoder inputs should be list, but % given" % type(decoder_inputs)
    with variable_scope.variable_scope("attention_decoder"):
        encoder_states = attention_keys
        batch_size = array_ops.shape(enc_padding_mask)[0]
        attn_size = encoder_states.get_shape()[2].value

        encoder_states = tf.expand_dims(encoder_states, axis=2)

        attention_vec_size = attn_size

        W_h = variable_scope.get_variable("W_h", [1, 1, attn_size, attention_vec_size])
        encoder_features = nn_ops.conv2d(encoder_states, W_h, [1, 1, 1, 1], "SAME")

        v = variable_scope.get_variable("v", [attention_vec_size])

        if use_coverage:
            with variable_scope.variable_scope("coverage"):
                w_c = variable_scope.get_variable("w_c", [1, 1, 1, attention_vec_size])

        if prev_coverage is not None:  # for beam search mode with coverage
            prev_coverage = tf.expand_dims(tf.expand_dims(prev_coverage, 2), 3)

        def attention(decoder_state, coverage=None):

            with variable_scope.variable_scope("Attention"):
                decoder_features = linear(decoder_state, attention_vec_size, True)
                decoder_features = tf.expand_dims(tf.expand_dims(decoder_features, 1), 1)

                def masked_attention(e):
                    """Take softmax of e then apply enc_padding_mask and re-normalize"""
                    e *= enc_padding_mask  # apply mask
                    attn_dist = nn_ops.softmax(e)  # take softmax. shape (batch_size, attn_length)
                    masked_sums = tf.reduce_sum(attn_dist, axis=1)  # shape (batch_size)
                    return attn_dist

                def sent_attn():
                    batch_nums = tf.range(0, limit=batch_size)
                    batch_nums = tf.expand_dims(batch_nums, 1)
                    attn_len = tf.shape(enc_sent_label)[1]
                    batch_nums = tf.tile(batch_nums, [1, attn_len])
                    sent_nums = tf.stack((batch_nums, enc_sent_label), axis=2)
                    masked_enc_features = encoder_features*tf.expand_dims(enc_padding_mask, axis=-1)
                    max_sent_num = tf.reduce_max(sent_nums)+1
                    shape = [batch_size, max_sent_num, tf.shape(encoder_features)[-1]]
                    sent_features = tf.scatter_nd(sent_nums, masked_enc_features, shape)
                    new_encoder_features = tf.gather(sent_nums)
                    e = math_ops.reduce_sum(
                        v * math_ops.tanh(new_encoder_features),
                        [2, 3])  # calculate e
                    attn_dist = nn_ops.softmax(e)

                if use_coverage and coverage is not None:
                    coverage_features = nn_ops.conv2d(coverage, w_c, [1, 1, 1, 1], "SAME")
                    e = math_ops.reduce_sum(
                        v * math_ops.tanh(
                            encoder_features +
                            decoder_features +
                            coverage_features), [2, 3])
                    attn_dist = masked_attention(e)
                    coverage += array_ops.reshape(attn_dist, [batch_size, -1, 1, 1])
                else:
                    e = math_ops.reduce_sum(
                        v * math_ops.tanh(encoder_features + decoder_features),
                        [2, 3])  # calculate e
                    attn_dist = masked_attention(e)

                    if use_coverage:  # first step of training
                        coverage = tf.expand_dims(tf.expand_dims(attn_dist, 2), 2)
                context_vector = math_ops.reduce_sum(
                    array_ops.reshape(attn_dist, [batch_size, -1, 1, 1]) * encoder_states, [1, 2])
                context_vector = array_ops.reshape(context_vector, [-1, attn_size])
            return context_vector, attn_dist, coverage

        outputs = []
        attn_dists = []
        state = initial_state
        coverage = prev_coverage
        context_vector = array_ops.zeros([batch_size, attn_size])
        context_vector.set_shape([None, attn_size])
        if initial_state_attention:  # true in decode mode
            context_vector, _, coverage = attention(initial_state, coverage)
        for i, inp in enumerate(decoder_inputs):
            if i > 0:
                variable_scope.get_variable_scope().reuse_variables()

            input_size = inp.get_shape().with_rank(2)[1]
            if input_size.value is None:
                raise ValueError("Could not infer input size from input: %s" % inp.name)
            input_size = inp.get_shape().as_list()[1]
            x = linear([inp] + [context_vector], input_size, True)

            cell_output, state = cell(x, state)

            if i == 0 and initial_state_attention:  # always true in decode mode
                with variable_scope.variable_scope(variable_scope.get_variable_scope(), reuse=True):
                    context_vector, attn_dist, _ = attention(state, coverage)
            else:
                context_vector, attn_dist, coverage = attention(state, coverage)
            attn_dists.append(attn_dist)

            with variable_scope.variable_scope("AttnOutputProjection"):
                output = linear([cell_output] + [context_vector], cell.output_size, True)
            outputs.append(output)

        if coverage is not None:
            coverage = array_ops.reshape(coverage, [batch_size, -1])

        return outputs, state, attn_dists


def conv_attention_decoder(emb_enc_inputs, enc_padding_mask, emb_dec_inputs, attentions_keys,
                           vocab_size, is_training, cnn_layers=4, nout_embed=256,
                           nhids_list=[256, 256, 256, 256], kwidths_list=[3, 3, 3, 3],
                           embedding_dropout_keep_prob=0.9, nhid_dropout_keep_prob=0.9,
                           out_dropout_keep_prob=0.9):
    """
    attentions_keys:
        a four dimensional tensor: (attention heads number, batch size, enc_length, hidden_dim)

    """

    enc_inputs = emb_enc_inputs
    input_shape = emb_dec_inputs.get_shape().as_list()    # static shape. may has None
    dec_labels = tf.contrib.layers.dropout(
        inputs=emb_dec_inputs,
        keep_prob=embedding_dropout_keep_prob,
        is_training=is_training)

    with tf.variable_scope("decoder_cnn"):
        next_layer = dec_labels
        if cnn_layers > 0:
            next_layer = linear_mapping_weightnorm(
                next_layer, nhids_list[0], dropout=embedding_dropout_keep_prob,
                var_scope_name="linear_mapping_before_cnn")

            next_layer, att_out, attn_dist = conv_decoder_stack(
                enc_inputs, dec_labels, attentions_keys, next_layer, enc_padding_mask,
                nhids_list, kwidths_list, is_training=is_training, dropout_dict={
                    'src': embedding_dropout_keep_prob,
                    'hid': nhid_dropout_keep_prob
                })

    with tf.variable_scope("softmax"):
        if is_training:
            next_layer = linear_mapping_weightnorm(next_layer, nout_embed, var_scope_name="linear_mapping_after_cnn")
        else:
            next_layer = linear_mapping_weightnorm(next_layer[:, -1:, :], nout_embed, var_scope_name="linear_mapping_after_cnn")
        outputs = tf.contrib.layers.dropout(
            inputs=next_layer,
            keep_prob=out_dropout_keep_prob,
            is_training=is_training)

    p_gens = linear_mapping_weightnorm(tf.concat(axis=-1, values=[outputs, att_out]), 1, 1, "p_gens")
    logits = linear_mapping_weightnorm(outputs, vocab_size, dropout=out_dropout_keep_prob, var_scope_name="logits_before_softmax")
    p_gens = tf.reshape(p_gens, [-1, input_shape[1], 1])
    logits = tf.reshape(logits, [-1, input_shape[1], vocab_size])

    return logits, p_gens, attn_dist, None, None
