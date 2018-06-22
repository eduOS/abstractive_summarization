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

    def __init__(self, tokens):
        self._tokens = tokens

    def __len__(self):
        return len(self.tokens)

    def extend(self, token):
        return Hypothesis(
            tokens=self._tokens + [token],
        )

    @property
    def latest_token(self):
        return self._tokens[-1]

    @property
    def tokens(self):
        return self._tokens


def run_monte_carlo_search(sess, model, vocab, batch, s_num=10):
    batch_size = model.hps.batch_size

    attention_keys, attention_values = model.run_encoder(sess, batch)
    stop_id = vocab.word2id(data.STOP_DECODING)
    pad_id = vocab.word2id(data.PAD_TOKEN)

    k_hyps = []
    batch_hyps = []
    for i in xrange(batch_size):
        hyps = [
            Hypothesis(
                tokens=[vocab.word2id(data.START_DECODING)],
            ) for _ in range(s_num)]
        batch_hyps.append(hyps)
    resample_num = 0
    min_dec_steps = 1
    for k in xrange(batch_size):
        hyps = batch_hyps[k]
        assert len(hyps) == s_num
        enc_padding_mask = np.tile(batch.enc_padding_mask[k], (s_num, 1))
        attention_key = np.tile(attention_keys[k], (s_num, 1, 1))
        attention_value = np.tile(attention_values[k], (s_num, 1, 1))
        steps = 0

        while steps < model.hps.max_dec_steps:
            dec_inputs = np.array([h.tokens for h in hyps])
            _, _, ran_id = model.run_decode_onestep(
                    sess, dec_inputs, attention_key, attention_value, enc_padding_mask)
            if steps < min_dec_steps and [stop_id] in ran_id.tolist():
                resample_num += 1
                continue
            steps += 1

            num_orig_hyps = len(hyps)
            _hyps = []
            for i in xrange(num_orig_hyps):
                h = hyps[i]
                n_token = ran_id[i][0] if h.latest_token not in [stop_id, pad_id] else pad_id
                new_hyp = h.extend(token=n_token)
                _hyps.append(new_hyp)

            hyps = _hyps
            if steps > min_dec_steps and [h.latest_token for h in hyps] == [pad_id for h in hyps]:
                break

        assert len(hyps) == s_num, colored("Hypothesis should be %s but given %s" % (s_num, len(hyps)), "red")
        k_hyps.append(hyps)

    if resample_num > (batch_size / 2):
        print(colored(
            "resampled %s times, the min_dec_steps is %s"
            % (resample_num, min_dec_steps), "red"))

    return attention_keys, k_hyps
