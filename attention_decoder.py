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

def lstm_attention_decoder(decoder_inputs, enc_padding_mask, attention_keys,
                           initial_state, cell, initial_state_attention=False,
                           use_coverage=False, prev_coverage=None):
    # can this be applied to beam repetitive batch?
    with variable_scope.variable_scope("attention_decoder"):
        encoder_states = attention_keys
        # if this line fails, it's because the batch size isn't defined
        batch_size = array_ops.shape(enc_padding_mask)[0]
        # if this line fails, it's because the attention length isn't defined
        attn_size = attention_keys.get_shape()[2].value

        # Reshape encoder_states (need to insert a dim)
        # now is shape (batch_size, attn_len, 1, attn_size)
        # the length is one
        encoder_states = tf.expand_dims(encoder_states, axis=2)

        # To calculate attention, we calculate
        #   v^T tanh(W_h h_i + W_s s_t + b_attn)
        # where h_i is an encoder state, and s_t a decoder state.
        # attn_vec_size is the length of the vectors v, b_attn, (W_h h_i) and
        # (W_s s_t).
        # We set it to be equal to the size of the encoder states.
        attention_vec_size = attn_size

        W_h = variable_scope.get_variable("W_h", [1, 1, attn_size, attention_vec_size])
        encoder_features = nn_ops.conv2d(encoder_states, W_h, [1, 1, 1, 1], "SAME")
        #  shape (batch_size, attn_length, 1, attention_vec_size)

        # Get the weight vectors v and w_c (w_c is for coverage)
        v = variable_scope.get_variable("v", [attention_vec_size])

        if use_coverage:
            with variable_scope.variable_scope("coverage"):
                w_c = variable_scope.get_variable("w_c", [1, 1, 1, attention_vec_size])

        if prev_coverage is not None:  # for beam search mode with coverage
            # reshape from (batch_size, attn_length) to (batch_size, attn_len,
            # 1, 1)
            prev_coverage = tf.expand_dims(tf.expand_dims(prev_coverage, 2), 3)

        def attention(decoder_state, coverage=None):
            """Calculate the context vector and attention distribution from the
            decoder state.

            Args:
              decoder_state: state of the decoder
              coverage: Optional. Previous timestep's coverage vector, shape
              (batch_size, attn_len, 1, 1).

            Returns:
              context_vector: weighted sum of encoder_states
              attn_dist: attention distribution
              coverage: new coverage vector. shape (batch_size, attn_len, 1, 1)
            """
            with variable_scope.variable_scope("Attention"):
                # Pass the decoder state through a linear layer (this is W_s s_t
                # + b_attn in the paper) shape (batch_size, attention_vec_size)
                decoder_features = linear(decoder_state, attention_vec_size, True)
                # reshape to (batch_size, 1, 1, attention_vec_size)
                decoder_features = tf.expand_dims(tf.expand_dims(decoder_features, 1), 1)
                # why not reshape?

                def masked_attention(e):
                    """Take softmax of e then apply enc_padding_mask and re-normalize"""
                    attn_dist = nn_ops.softmax(e)  # take softmax. shape (batch_size, attn_length)
                    e *= enc_padding_mask  # apply mask
                    masked_sums = tf.reduce_sum(attn_dist, axis=1)  # shape (batch_size)
                    return attn_dist / tf.reshape(masked_sums, [-1, 1])  # re-normalize

                if use_coverage and coverage is not None:
                    # non-first step of coverage
                    # Multiply coverage vector by w_c to get coverage_features.
                    # c has shape (batch_size, attn_length, 1,
                    # attention_vec_size)
                    coverage_features = nn_ops.conv2d(coverage, w_c, [1, 1, 1, 1], "SAME")

                    # Calculate v^T tanh(W_h h_i + W_s s_t + w_c c_i^t + b_attn)
                    # shape (batch_size,attn_length)
                    e = math_ops.reduce_sum(
                        v * math_ops.tanh(
                            encoder_features +
                            decoder_features +
                            coverage_features), [2, 3])

                    # (batch_size, 1,           1, attention_vec_size)
                    # (batch_size, attn_length, 1, attention_vec_size)
                    # (batch_size, 1,           1, attention_vec_size)
                    # (batch_size, attn_length, 1, attention_vec_size)

                    # Take softmax of e to get the attention distribution
                    # shape (batch_size, attn_length)
                    attn_dist = masked_attention(e)

                    # Update coverage vector, the initial coverage is zero
                    coverage += array_ops.reshape(attn_dist, [batch_size, -1, 1, 1])
                else:
                    # Calculate v^T tanh(W_h h_i + W_s s_t + b_attn)
                    e = math_ops.reduce_sum(
                        v * math_ops.tanh(encoder_features + decoder_features),
                        [2, 3])  # calculate e

                    # Take softmax of e to get the attention distribution
                    # shape (batch_size, attn_length)
                    attn_dist = masked_attention(e)

                    if use_coverage:  # first step of training
                        coverage = tf.expand_dims(tf.expand_dims(attn_dist, 2), 2)
                        # shape (batch_size, attn_length, 1, 1)
                        # initialize coverage

                # Calculate the context vector from attn_dist and encoder_states
                context_vector = math_ops.reduce_sum(
                    array_ops.reshape(attn_dist, [batch_size, -1, 1, 1]) * encoder_states, [1, 2])
                # shape (batch_size, attn_size).
                context_vector = array_ops.reshape(context_vector, [-1, attn_size])

            return context_vector, attn_dist, coverage

        outputs = []
        attn_dists = []
        p_gens = []
        state = initial_state
        coverage = prev_coverage
        # initialize coverage to None or whatever was passed in
        context_vector = array_ops.zeros([batch_size, attn_size])
        # Ensure the second shape of attention vectors is set.
        context_vector.set_shape([None, attn_size])
        if initial_state_attention:  # true in decode mode
            # Re-calculate the context vector from the previous step so that we
            # can pass it through a linear layer with this step's input to get a
            # modified version of the input
            # in decode mode, this is what updates the coverage vector
            context_vector, _, coverage = attention(initial_state, coverage)
        for i, inp in enumerate(decoder_inputs):
            # when should this terminate due to beam size
            if i > 0:
                variable_scope.get_variable_scope().reuse_variables()

            # Merge input and previous attentions into one vector x of the same
            # size as inp
            input_size = inp.get_shape().with_rank(2)[1]
            if input_size.value is None:
                raise ValueError("Could not infer input size from input: %s" % inp.name)
            input_size = inp.get_shape().as_list()[1]
            x = linear([inp] + [context_vector], input_size, True)
            # is this the same in either mode?
            # only for the training, while decoding is is the beam search

            # Run the decoder RNN cell. cell_output = decoder state
            cell_output, state = cell(x, state)
            # state is the h_i^{d}. e_{ti}

            # Run the attention mechanism.
            if i == 0 and initial_state_attention:  # always true in decode mode
                # you need this because you've already run the initial
                # attention(...) call
                with variable_scope.variable_scope(variable_scope.get_variable_scope(), reuse=True):
                    context_vector, attn_dist, _ = attention(state, coverage)
                    # don't allow coverage to update
            else:
                context_vector, attn_dist, coverage = attention(state, coverage)
            attn_dists.append(attn_dist)

            # Calculate p_gen
            with tf.variable_scope('calculate_pgen'):
                p_gen = linear([context_vector, state.c, state.h, x], 1, True)
                # a scalar
                # p_gen = maxout(p_gen, 1)
                # p_gen = tf.reshape(p_gen, 1)
                p_gen = tf.sigmoid(p_gen)
                p_gens.append(p_gen)

            # Concatenate the cell_output (= decoder state) and the context
            # vector, and pass them through a linear layer
            # This is V[s_t, h*_t] + b in the paper
            with variable_scope.variable_scope("AttnOutputProjection"):
                output_2 = linear([cell_output] + [context_vector], cell.output_size * 2, True)
                output = maxout(output_2, cell.output_size)
                output = tf.reshape(output, [-1, cell.output_size])
            outputs.append(output)

        # If using coverage, reshape it
        if coverage is not None:
            coverage = array_ops.reshape(coverage, [batch_size, -1])

        return outputs, p_gens, attn_dists, state, coverage


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

            next_layer, att_out, attn_dists = conv_decoder_stack(
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

    # outputs, att_out, attn_dists = conv_block(inputs, enc_states, attention_states, vocab_size, True)
    p_gens = linear_mapping_weightnorm(tf.concat(axis=-1, values=[outputs, att_out]), 1, 1, "p_gens")
    logits = linear_mapping_weightnorm(outputs, vocab_size, dropout=out_dropout_keep_prob, var_scope_name="logits_before_softmax")
    # reshape for the length to unstack
    p_gens = tf.reshape(p_gens, [-1, input_shape[1], 1])
    logits = tf.reshape(logits, [-1, input_shape[1], vocab_size])

    return logits, p_gens, attn_dists, None, None
