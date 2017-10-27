# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function
from __future__ import absolute_import
from __future__ import division
import tensorflow as tf


# beam decoder
def beam_dec(length,
             initial_state,
             input_embedding,
             cell,
             logit_fn,
             num_candidates=1,
             beam_size=100,
             gamma=0.65):
    """ A basic beam decoder

    """

    batch_size = tf.shape(initial_state[0])[0] \
        if isinstance(initial_state, tuple) else \
        tf.shape(initial_state)[0]
    inputs_size = input_embedding.get_shape()[1].value
    # the time steps
    inputs = tf.nn.embedding_lookup(
        input_embedding, tf.zeros([batch_size], dtype=tf.int32))
    vocab_size = tf.shape(input_embedding)[0]

    # iter
    outputs, state = cell(inputs, initial_state)
    logits = logit_fn(outputs)

    prev = tf.nn.log_softmax(logits)
    probs = tf.slice(prev, [0, 1], [-1, -1])
    # copy except the first column
    best_probs, indices = tf.nn.top_k(probs, beam_size)

    symbols = indices % vocab_size + 1
    beam_parent = indices // vocab_size
    beam_parent = tf.reshape(
        tf.expand_dims(tf.range(batch_size), 1) + beam_parent, [-1])
    paths = tf.reshape(symbols, [-1, 1])

    candidates = [tf.to_int32(tf.zeros([batch_size, 1, length]))]
    scores = [tf.slice(prev, [0, 0], [-1, 1])]

    tf.get_variable_scope().reuse_variables()

    for i in range(length-1):

        if isinstance(state, tuple):
            state = tuple([tf.gather(s, beam_parent) for s in state])
        else:
            state = tf.gather(state, beam_parent)

        inputs = tf.reshape(tf.nn.embedding_lookup(input_embedding, symbols),
                            [-1, inputs_size])

        # iter
        outputs, state = cell(inputs, state)
        logits = logit_fn(outputs)

        prev = tf.reshape(tf.nn.log_softmax(logits),
                          [batch_size, beam_size, vocab_size])

        # add the path and score of the candidates in the current beam to the lists
        fn = lambda seq: tf.size(tf.unique(seq)[0])
        uniq_len = tf.reshape(
            tf.to_float(tf.map_fn(
                fn, paths, dtype=tf.int32, parallel_iterations=100000)),
            [batch_size, beam_size])
        close_score = best_probs / (uniq_len ** gamma) + tf.squeeze(
            tf.slice(prev, [0, 0, 0], [-1, -1, 1]), [2])
        candidates.append(tf.reshape(tf.pad(paths,
                                            [[0, 0], [0, length-1-i]],
                                            "CONSTANT"),
                                     [batch_size, beam_size, length]))
        scores.append(close_score)

        prev += tf.expand_dims(best_probs, 2)
        probs = tf.reshape(
            tf.slice(prev, [0, 0, 1], [-1, -1, -1]), [batch_size, -1])
        best_probs, indices = tf.nn.top_k(probs, beam_size)

        symbols = indices % (vocab_size - 1) + 1
        beam_parent = indices // (vocab_size - 1)
        beam_parent = tf.reshape(
            tf.expand_dims(tf.range(batch_size) * beam_size, 1) + beam_parent,
            [-1])
        paths = tf.gather(paths, beam_parent)
        paths = tf.concat([paths, tf.reshape(symbols, [-1, 1])], 1)

    # pick the topk from the candidates in the lists
    candidates = tf.reshape(tf.concat(candidates, 1), [-1, length])
    scores = tf.concat(scores, 1)
    best_scores, indices = tf.nn.top_k(scores, num_candidates)
    indices = tf.reshape(
        tf.expand_dims(
            tf.range(batch_size) * (beam_size * (length-1) + 1), 1) + indices, [-1])
    best_candidates = tf.reshape(tf.gather(candidates, indices),
                                 [batch_size, num_candidates, length])

    return best_candidates, best_scores
