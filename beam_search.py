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

"""This file contains code to run beam search decoding"""
from __future__ import unicode_literals, print_function
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
import data
from codecs import open

FLAGS = tf.app.flags.FLAGS


class Hypothesis(object):
    """Class to represent a hypothesis during beam search. Holds all the
    information needed for the hypothesis."""

    def __init__(self, tokens, log_probs, state, attn_dists, p_gens, coverage):
        """Hypothesis constructor.

        Args:
          tokens: List of integers. The ids of the tokens that form the summary
          so far.
          log_probs: List, same length as tokens, of floats, giving the log
          probabilities of the tokens so far.
          state: Current state of the decoder, a LSTMStateTuple.
          attn_dists: List, same length as tokens, of numpy arrays with shape
          (attn_length). These are the attention distributions so far.
          p_gens: List, same length as tokens, of floats, or None if not using
          pointer-generator model. The values of the generation probability so
          far.
          coverage: Numpy array of shape (attn_length), or None if not using
          coverage. The current coverage vector.
        """
        self.tokens = tokens
        self.log_probs = log_probs
        self.state = state
        self.attn_dists = attn_dists
        self.p_gens = p_gens
        self.coverage = coverage

    def extend(self, token, log_prob, state, attn_dist, p_gen, coverage):
        """Return a NEW hypothesis, extended with the information from the
        latest step of beam search.

        Args:
          token: Integer. Latest token produced by beam search.
          log_prob: Float. Log prob of the latest token.
          state: Current decoder state, a LSTMStateTuple.
          attn_dist: Attention distribution from latest step. Numpy array shape
          (attn_length).
          p_gen: Generation probability on latest step. Float.
          coverage: Latest coverage vector. Numpy array shape (attn_length), or
          None if not using coverage.
        Returns:
          New Hypothesis for next step.
        """
        # can I use the class method here?
        return Hypothesis(
            tokens=self.tokens + [token],
            log_probs=self.log_probs + [log_prob],
            state=state,
            attn_dists=self.attn_dists + [attn_dist],
            p_gens=self.p_gens + [p_gen],
            coverage=coverage
        )

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def log_prob(self):
        # the log probability of the hypothesis so far is the sum of the log
        # probabilities of the tokens so far
        return sum(self.log_probs)

    @property
    def avg_log_prob(self):
        # normalize log probability by number of tokens (otherwise longer
        # sequences always have lower probability)
        # why?
        return self.log_prob / len(self.tokens)


def run_greedy_search(sess, model, vocab, batch):
    enc_states, dec_in_state = model.run_encoder(sess, batch)
    latest_tokens = [vocab.word2id(data.START_DECODING)]*FLAGS.batch_size
    latest_tokens = np.transpose(np.array([latest_tokens]))
    steps = 0
    greedy_outcome = []
    prev_coverage = np.zeros([batch.enc_batch.shape[1]])
    print("MARK")
    while(steps < FLAGS.max_dec_steps):
        (
            topk_ids, topk_log_probs, new_states,
            attn_dists, p_gens, prev_coverage
        ) = model.run_decode_onestep(
            sess=sess, enc_batch_extend_vocab=np.array(batch.enc_batch_extend_vocab),
            max_art_oovs=batch.max_art_oovs, latest_tokens=latest_tokens,
            enc_states=enc_states, enc_padding_mask=np.array(batch.enc_padding_mask),
            dec_init_states=dec_in_state, prev_coverage=prev_coverage,
        )
        print(new_states.c.shape)
        print(new_states.h.shape)
        latest_tokens = [t if t in range(vocab.size()) else vocab.word2id(data.UNKNOWN_TOKEN) for t in topk_ids[:, 0].tolist()]
        latest_tokens = np.transpose(np.array([latest_tokens]))
        greedy_outcome.append([vocab.id2word(i) for i in topk_ids[:, 0].tolist()])
        dec_in_state = new_states
        steps += 1

    with open("tem.txt", 'w', "utf-8") as f:
        for ok in greedy_outcome:
            f.write(" ".join(ok) + "\n")

def sort_hyps(hyps):
    """Return a list of Hypothesis objects, sorted by descending average log
    probability"""
    return sorted(hyps, key=lambda h: h.avg_log_prob, reverse=True)
