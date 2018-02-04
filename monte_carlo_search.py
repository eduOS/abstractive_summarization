# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
from termcolor import colored
import data
from six.moves import xrange

FLAGS = tf.app.flags.FLAGS


class Hypothesis(object):
    """Class to represent a hypothesis during beam search. Holds all the
    information needed for the hypothesis."""

    def __init__(self, tokens, state, coverage):
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
        self._tokens = tokens
        self.state = state
        self.coverage = coverage

    def __len__(self):
        return len(self.tokens)

    def extend(self, token, state, coverage):
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
        if not token:
            # if the last token is stop decoding then don't update the
            # hypothesis
            return Hypothesis(
                tokens=self._tokens,
                state=self.state,
                coverage=self.coverage
            )
        return Hypothesis(
            tokens=self._tokens + [token],
            state=state,
            coverage=coverage
        )

    @property
    def latest_token(self):
        return self._tokens[-1]

    @property
    def tokens(self):
        return self._tokens


def run_monte_carlo_search(sess, model, vocab, batch, s_num=10):
    batch_size = model.hps.batch_size

    enc_states, dec_in_state = model.run_encoder(sess, batch)
    stop_id = vocab.word2id(data.STOP_DECODING)
    k_hyps = []
    batch_hyps = []
    # seperated hyps for each beam
    for i in xrange(batch_size):
        hyps = [
            Hypothesis(
                tokens=[vocab.word2id(data.START_DECODING)],
                state=tf.contrib.rnn.LSTMStateTuple(
                    dec_in_state.c[i], dec_in_state.h[i]),
                coverage=np.zeros([batch.enc_batch.shape[1]])
            ) for _ in range(s_num)]
        batch_hyps.append(hyps)
    # this will contain finished hypotheses (those that have emitted the
    # [STOP] token)

    # this can be optimized into multithread
    resample_num = 0
    min_dec_steps = 1
    for k in xrange(batch_size):
        hyps = batch_hyps[k]
        assert len(hyps) == s_num
        enc_batch_extend_vocab = np.tile(batch.enc_batch_extend_vocab[k], (s_num, 1))
        enc_padding_mask = np.tile(batch.enc_padding_mask[k], (s_num, 1))
        enc_states_ = np.tile(enc_states[k], (s_num, 1, 1))
        steps = 0

        while steps < model.hps.max_dec_steps:
            # latest token produced by each hypothesis
            latest_tokens = [h.latest_token for h in hyps]
            if latest_tokens == [stop_id for h in hyps]:
                break
            # change any in-article temporary OOV ids to [UNK] id, so that we can
            # lookup word embeddings
            latest_tokens = [
                t if t in xrange(
                    vocab.size()) else vocab.word2id(data.UNKNOWN_TOKEN)
                for t in latest_tokens]
            latest_tokens = np.transpose(np.array([latest_tokens]))
            # UNKNOWN_TOKEN will be replaced with a placeholder
            # max_art_oovs = np.tile(batch.max_art_oovs[k], (beam_size, 1))
            # list of current decoder states of the hypotheses
            states = [h.state for h in hyps]
            # list of coverage vectors (or None)
            prev_coverage = np.stack([h.coverage for h in hyps], axis=0)

            # Run one step of the decoder to get the new info, it is the same either
            # in decoding or in gan

            ran_id, _, new_states, _, _, new_coverage = model.run_decode_onestep(
                sess=sess, enc_batch_extend_vocab=enc_batch_extend_vocab,
                max_art_oovs=batch.max_art_oovs, latest_tokens=latest_tokens,
                enc_states=enc_states_, enc_padding_mask=enc_padding_mask,
                dec_init_states=states, prev_coverage=prev_coverage,
                method="mc"
            )
            if steps < min_dec_steps and [stop_id] in ran_id.tolist():
                resample_num += 1
                continue
            steps += 1

            num_orig_hyps = len(hyps)
            _hyps = []
            for i in xrange(num_orig_hyps):
                h, new_state, new_coverage_i = hyps[i], new_states[i], new_coverage[i]
                # take the ith hypothesis and new decoder state info
                # for each of the top 2*beam_size hyps:
                n_token = ran_id[i][0] if h.latest_token != stop_id else None
                new_hyp = h.extend(
                    token=n_token,
                    state=new_state,
                    coverage=new_coverage_i)
                _hyps.append(new_hyp)

            hyps = _hyps

        assert len(hyps) == s_num, colored("Hypothesis should be %s but given %s" % (s_num, len(hyps)), "red")
        k_hyps.append(hyps)

    if resample_num > (batch_size / 2):
        print(colored(
            "resampled %s times, the min_dec_steps is %s"
            % (resample_num, min_dec_steps), "red"))

    # Return the hypothesis with highest average log prob
    return enc_states, dec_in_state, k_hyps
