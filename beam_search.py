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
from six.moves import xrange

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
        """
        self._tokens = tokens
        self.log_probs = log_probs
        self.state = state
        self.attn_dists = attn_dists

    def __len__(self):
        return len(self._tokens)

    def extend(self, token, log_prob, state, attn_dist):
        """Return a NEW hypothesis, extended with the information from the
        latest step of beam search.

        Args:
          token: Integer. Latest token produced by beam search.
          log_prob: Float. Log prob of the latest token.
          state: Current decoder state, a LSTMStateTuple.
          attn_dist: Attention distribution from latest step. Numpy array shape
          (attn_length).
          p_gen: Generation probability on latest step. Float.
        Returns:
          New Hypothesis for next step.
        """
        # can I use the class method here?
        return Hypothesis(
            tokens=self._tokens + [token],
            log_probs=self.log_probs + [log_prob],
            state=state,
            attn_dists=self.attn_dists + [attn_dist],
        )

    @property
    def latest_token(self):
        return self._tokens[-1]

    @property
    def log_prob(self):
        # the log probability of the hypothesis so far is the sum of the log
        # probabilities of the tokens so far
        return sum(self.log_probs)

    @property
    def tokens(self):
        return self._tokens

    @property
    def avg_log_prob(self):
        # normalize log probability by number of tokens (otherwise longer
        # sequences always have lower probability)
        # why?
        return self.log_prob / len(self._tokens)


def run_beam_search(sess, model, vocab, batch, top_k=1):
    """ For the GAN
    Performs beam search decoding on the given example.

    Args:
      sess: a tf.Session
      model: a seq2seq model
      vocab: Vocabulary object
      batch: Batch object that has the beam_size * batch_size samples,
        batch_size kinds of sample, each sample repeated beam_size times
        What actually matters in the decode_onestep is the max_art_oovs and
        enc_batch_extend_vocab

    Returns:
        enc_states: one of the outputs of the encoder which is of shape [batch_size, length],
        dec_in_state: one of the outputs of the encoder which is of shape [batch_size, 2*hidden_dim],
        best_k_hyp: Hypothesis object; the best k hypothesis found by beam search.
    """
    batch_size = model.hps.batch_size
    beam_size = FLAGS.beam_size
    if top_k > beam_size:
        top_k = beam_size
    # Run the encoder to get the encoder hidden states and decoder initial state
    # dec_in_state is a LSTMStateTuple
    # enc_states has shape [batch_size, <=max_enc_steps, 2*hidden_dim].
    enc_states, dec_in_state = model.run_encoder(sess, batch)
    # enc_states and dec_in_state should be scaled to match the latter setting

    best_k_hyps = []
    batch_hyps = []
    # seperated hyps for each beam
    for i in xrange(batch_size):
        hyps = [
            Hypothesis(
                tokens=[vocab.word2id(data.START_DECODING)],
                log_probs=[0.0],
                state=tf.contrib.rnn.LSTMStateTuple(
                    dec_in_state.c[i], dec_in_state.h[i]),
                attn_dists=[],
                # zero vector of length attention_length
            ) for _ in range(beam_size)]
        batch_hyps.append(hyps)
    # this will contain finished hypotheses (those that have emitted the
    # [STOP] token)

    # this can be optimized into multithread
    for k in xrange(batch_size):
        hyps = batch_hyps[k]
        enc_batch_extend_vocab = np.tile(batch.enc_batch_extend_vocab[k], (beam_size, 1))
        enc_padding_mask = np.tile(batch.enc_padding_mask[k], (beam_size, 1))
        enc_states_ = np.tile(enc_states[k], (beam_size, 1, 1))
        results = []
        steps = 0

        while steps < model.hps.max_dec_steps and len(results) < beam_size:
            # latest token produced by each hypothesis
            latest_tokens = [h.latest_token for h in hyps]
            # change any in-article temporary OOV ids to [UNK] id, so that we can
            # lookup word embeddings
            latest_tokens = [
                t if t in xrange(vocab.size()) else vocab.word2id(data.UNKNOWN_TOKEN)
                for t in latest_tokens]
            latest_tokens = np.transpose(np.array([latest_tokens]))
            # UNKNOWN_TOKEN will be replaced with a placeholder
            # max_art_oovs = np.tile(batch.max_art_oovs[k], (beam_size, 1))
            # list of current decoder states of the hypotheses
            states = [h.state for h in hyps]
            # Run one step of the decoder to get the new info, it is the same either
            # in decoding or in gan

            topk_ids, topk_log_probs, new_states, attn_dists \
                = model.run_decode_onestep(
                    sess=sess, enc_batch_extend_vocab=enc_batch_extend_vocab,
                    max_art_oovs=batch.max_art_oovs, latest_tokens=latest_tokens,
                    enc_states=enc_states_, enc_padding_mask=enc_padding_mask,
                    dec_init_states=states
                )

            # Extend each hypothesis and collect them all in all_hyps
            all_hyps = []
            # On the first step, we only had one original hypothesis (the initial
            # hypothesis). On subsequent steps, all original hypotheses are
            # distinct.
            num_orig_hyps = 1 if steps == 0 else len(hyps)
            for i in xrange(num_orig_hyps):
                h, new_state, attn_dist = (
                    hyps[i], new_states[i], attn_dists[i])
                # take the ith hypothesis and new decoder state info
                # for each of the top 2*beam_size hyps:
                for j in range(beam_size * 2):
                    # Extend the ith hypothesis with the jth option
                    new_hyp = h.extend(
                        token=topk_ids[i, j],
                        log_prob=topk_log_probs[i, j],
                        state=new_state,
                        attn_dist=attn_dist)
                    all_hyps.append(new_hyp)

            # Filter and collect any hypotheses that have produced the end token.
            hyps = []  # will contain hypotheses for the next step
            for h in sort_hyps(all_hyps):  # in order of most likely h
                if h.latest_token == vocab.word2id(data.STOP_DECODING):
                    # if stop token is reached...
                    # If this hypothesis is sufficiently long, put in results.
                    # Otherwise discard.
                    if steps >= model.hps.min_dec_steps:
                        results.append(h)
                else:
                    # hasn't reached stop token, so continue to extend this
                    # hypothesis
                    hyps.append(h)
                if len(hyps) == beam_size or len(results) == beam_size:
                    # Once we've collected beam_size-many hypotheses for the next
                    # step, or beam_size-many complete hypotheses, stop.
                    # print('length of hyps or results is equal to beam_size, break.')
                    break

            steps += 1

        # At this point, either we've got beam_size results, or we've reached
        # maximum decoder steps

        if len(results) < top_k:
            # if we don't have any complete results, add all current hypotheses
            # (incomplete summaries) to results
            results += hyps[:top_k - len(results)]

        # Sort hypotheses by average log probability
        hyps_sorted = sort_hyps(results)
        if top_k == 1:
            best_k_hyp = hyps_sorted[0]
        else:
            best_k_hyp = hyps_sorted[:top_k]
        best_k_hyps.append(best_k_hyp)

    # Return the hypothesis with highest average log prob
    return enc_states, dec_in_state, best_k_hyps


def sort_hyps(hyps):
    """Return a list of Hypothesis objects, sorted by descending average log
    probability"""
    return sorted(hyps, key=lambda h: h.avg_log_prob, reverse=True)
