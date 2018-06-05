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

    def __init__(self, tokens, log_probs):
        self._tokens = tokens
        self.log_probs = log_probs

    def __len__(self):
        return len(self._tokens)

    def extend(self, token, log_prob):
        return Hypothesis(
            tokens=self._tokens + [token],
            log_probs=self.log_probs + [log_prob],
        )

    @property
    def log_prob(self):
        return sum(self.log_probs)

    @property
    def tokens(self):
        return self._tokens

    @property
    def latest_token(self):
        return self._tokens[-1]

    @property
    def avg_log_prob(self):
        return self.log_prob / len(self._tokens)


def run_beam_search(sess, model, vocab, batch, top_k=1):
    batch_size = model.hps.batch_size
    beam_size = FLAGS.beam_size
    if top_k > beam_size:
        top_k = beam_size
    attention_keys, attention_values = model.run_encoder(sess, batch)

    best_k_hyps = []
    batch_hyps = []
    for i in xrange(batch_size):
        hyps = [
            Hypothesis(
                tokens=[vocab.word2id(data.START_DECODING)],
                log_probs=[0.0],
            ) for _ in range(beam_size)]
        batch_hyps.append(hyps)

    for k in xrange(batch_size):
        hyps = batch_hyps[k]
        enc_padding_mask = np.tile(batch.enc_padding_mask[k], (beam_size, 1))
        attention_key = np.tile(attention_keys[k], (beam_size, 1, 1))
        attention_value = np.tile(attention_values[k], (beam_size, 1, 1))
        results = []
        steps = 0

        while steps < model.hps.max_dec_steps and len(results) < beam_size:
            dec_inputs = np.array([h.tokens for h in hyps])
            topk_log_probs, topk_ids = model.run_decode_onestep(
                    sess, dec_inputs, attention_key, attention_value, enc_padding_mask,
                )

            all_hyps = []
            num_orig_hyps = 1 if steps == 0 else len(hyps)
            for i in xrange(num_orig_hyps):
                h = hyps[i]
                for j in range(beam_size * 2):
                    new_hyp = h.extend(
                        token=topk_ids[i, j],
                        log_prob=topk_log_probs[i, j]
                    )
                    all_hyps.append(new_hyp)

            hyps = []
            for h in sort_hyps(all_hyps):
                if h.latest_token == vocab.word2id(data.STOP_DECODING):
                    if steps >= model.hps.min_dec_steps:
                        results.append(h)
                else:
                    hyps.append(h)
                if len(hyps) == beam_size or len(results) == beam_size:
                    break

            steps += 1

        if len(results) < top_k:
            results += hyps[:top_k - len(results)]

        hyps_sorted = sort_hyps(results)
        if top_k == 1:
            best_k_hyp = hyps_sorted[0]
        else:
            best_k_hyp = hyps_sorted[:top_k]
        best_k_hyps.append(best_k_hyp)

    return best_k_hyps


def sort_hyps(hyps):
    """Return a list of Hypothesis objects, sorted by descending average log
    probability"""
    return sorted(hyps, key=lambda h: h.avg_log_prob, reverse=True)
