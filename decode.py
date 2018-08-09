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

"""This file contains code to run beam search decoding, including running ROUGE
evaluation and producing JSON datafiles for the in-browser attention visualizer,
which can be found here https://github.com/abisee/attn_vis"""
from __future__ import unicode_literals, print_function
from __future__ import absolute_import
from __future__ import division

import os
from gan_utils import rouge_l
from data import strip_pads
import math
import time
from data import outputsids2words
from data import pad_equal_length
import tensorflow as tf
from random import randint
import beam_search
import monte_carlo_search
import data
import json
from codecs import open
# import pyrouge
# import gen_utils
import numpy as np
import logging
from six.moves import xrange
from data import PAD_TOKEN, STOP_DECODING
# import numpy as np
FLAGS = tf.app.flags.FLAGS

SECS_UNTIL_NEW_CKPT = 60  # max number of seconds before loading new checkpoint


class Decoder(object):
    """Beam search decoder."""

    def __init__(self, sess, model, vocab):
        """Initialize decoder.

        Args:
          model: a Seq2SeqAttentionModel object.
          batcher: a Batcher object.
          vocab: Vocabulary object
        """
        self._model = model
        # self._model.build_graph()
        self._vocab = vocab
        # we use this to load checkpoints for decoding
        self._sess = sess
        self._hps = model.hps

    def prepare_dir(self):
        ckpt_name = str(time.time())[:10]
        self._decode_dir = os.path.join(
            self._hps.log_root, get_decode_dir_name(self._hps, ckpt_name))
        if os.path.exists(self._decode_dir):
            raise Exception(
                "single_pass decode directory %s should not already exist" %
                self._decode_dir)

        # Make the decode dir if necessary
        if not os.path.exists(self._decode_dir):
            os.mkdir(self._decode_dir)

        # Make the dirs to contain output written in the correct format for
        # pyrouge
        self._rouge_ref_dir = os.path.join(self._decode_dir, "reference")
        if not os.path.exists(self._rouge_ref_dir):
            os.mkdir(self._rouge_ref_dir)
        self._rouge_dec_dir = os.path.join(self._decode_dir, "decoded")
        if not os.path.exists(self._rouge_dec_dir):
            os.mkdir(self._rouge_dec_dir)

    def mc_generate(self, batch, s_num=4):
        # Run beam search to get best Hypothesis
        attention_keys, n_hyps_batch = monte_carlo_search.run_monte_carlo_search(
            self._sess, self._model, self._vocab, batch, s_num=s_num)

        padded_n_hyps = []
        pad_id = self._vocab.word2id(PAD_TOKEN)
        stop_id = self._vocab.word2id(STOP_DECODING)
        padding_max_len = self._hps.max_dec_steps
        sample_max_len = self._hps.max_dec_steps + 1
        padding_mask = np.zeros((len(n_hyps_batch), s_num, padding_max_len), dtype=np.int32)
        for b, n_hyps in enumerate(n_hyps_batch):
            padded_hyps = []
            for n, hyp in enumerate(n_hyps):
                tokens = hyp.tokens
                length_exclude_start_token = tokens.index(stop_id) if stop_id in tokens else len(tokens) - 1
                padding_mask[b, n, :length_exclude_start_token] = 1
                padded = tokens + (sample_max_len - len(hyp)) * [pad_id] if len(hyp) < sample_max_len else tokens[:sample_max_len]
                assert len(padded) == sample_max_len, "sample should be of length %s, but %s given." % (sample_max_len, len(padded))
                padded_hyps.append(padded)
            padded_n_hyps.append(padded_hyps)

        outputs_ids = np.array(padded_n_hyps).astype(int)

        # transfer to (s_num, batch_size, max_dec_steps)
        outputs_ids = [np.squeeze(i, 1) for i in np.split(outputs_ids, outputs_ids.shape[1], 1)]
        padding_mask = [
            np.squeeze(i, 1)
            for i in np.split(padding_mask, padding_mask.shape[1], 1)]
        assert len(outputs_ids) == s_num

        return attention_keys, outputs_ids, padding_mask

    def multinomial_decode(self, sess, model, batch, vocab):
        batch_size = len(batch.enc_batch_extend_vocab)
        ran_ids = []
        id_mappings = []
        enc_states, dec_in_state = model.run_encoder(sess, batch)
        latest_tokens = batch_size * [vocab.word2id(data.START_DECODING)]
        prev_coverage = np.zeros([batch.enc_batch.shape[1]])
        dec_state = dec_in_state

        steps = 0
        while steps < self._hps.max_dec_steps:
            ran_ids.append(latest_tokens)
            latest_tokens = [
                t if t in xrange(
                    vocab.size()) else vocab.word2id(data.UNKNOWN_TOKEN)
                for t in latest_tokens]
            latest_tokens = np.transpose(np.array([latest_tokens]))
            ran_id, _, dec_state, _, _, prev_coverage = model.run_decode_onestep(
                sess=sess, enc_batch_extend_vocab=batch.enc_batch_extend_vocab,
                max_art_oovs=batch.max_art_oovs, latest_tokens=latest_tokens,
                enc_states=enc_states, enc_padding_mask=batch.enc_padding_mask,
                dec_init_states=dec_state, prev_coverage=prev_coverage,
                method="mc"
            )
            latest_tokens = ran_id.tolist()

        stop_id = vocab.word2id(data.STOP_DECODING)
        for ran_id in ran_ids:
            m = ran_id.index(stop_id)
            if m:
                id_mapping = [1] * (m+1) + (len(ran_id) - m - 1) * [0]
            else:
                id_mapping = [1] * len(ran_id)
            id_mappings.append(id_mapping)

        return enc_states, dec_in_state, np.array(ran_ids), np.array(id_mappings)

    def bs_decode(self, sess, discriminator, batcher, save2file=True, single_pass=True, sample_rate=0):
        """Decode examples until data is exhausted (if self._hps.single_pass) and
        return, or decode indefinitely, loading latest checkpoint at regular
        intervals"""

        rouge_scores = []
        dis_rouge_scores = []
        max_dis_rouge_scores = []
        if save2file:
            self.prepare_dir()
            ref_file = os.path.join(
                self._rouge_ref_dir, "reference.txt")
            decoded_file = os.path.join(
                self._rouge_dec_dir, "decoded.txt")
            overview_file = os.path.join(
                self._decode_dir, "overview.txt")
            ref_f = open(ref_file, "a", 'utf-8')
            dec_f = open(decoded_file, "a", 'utf-8')
            ove_f = open(overview_file, "a", 'utf-8')

        batch_size = self._hps.batch_size

        counter = 0
        try:
            while True:
                batch = batcher.next_batch()
                if batch is None:
                    assert single_pass, (
                        "Dataset exhausted, but we are not in single_pass mode")
                    print("Decoder has finished reading dataset for single_pass.")
                    average_rouge = np.mean(np.array(rouge_scores))
                    dis_average_rouge = np.mean(np.array(dis_rouge_scores))
                    max_dis_average_rouge = np.mean(np.array(max_dis_rouge_scores))
                    if save2file:
                        ref_f.close()
                        dec_f.close()
                        ove_f.write("\nThe overall average rouge: %s" % average_rouge)
                        ove_f.write("\nThe overall average rouge according to dis: %s" % dis_average_rouge)
                        ove_f.write("\nThe overall average rouge according to max dis: %s" % max_dis_average_rouge)
                        ove_f.close()
                        return average_rouge
                    else:
                        return average_rouge

                best_hyps = beam_search.run_beam_search(self._sess, self._model, self._vocab, batch)[0]
                outputs_ids = [[t for t in hyp.tokens[1:]] for hyp in best_hyps]
                padded_outputs_ids = pad_equal_length(
                    outputs_ids, self._vocab.word2id(STOP_DECODING),
                    self._vocab.word2id(PAD_TOKEN), self._hps.max_dec_steps)
                # the probs for each sample by the generator distribution
                sample_mean_generator_probs = [hyp.prob for hyp in best_hyps]

                original_articles = batch.original_articles
                original_abstracts = batch.original_abstracts
                article_lens = np.tile(batch.enc_lens, len(outputs_ids))
                articles = np.tile(batch.enc_batch, (len(outputs_ids), 1))
                # TODO: article and article_lens should be expended to the some
                # shape as padded_outputs_ids
                emb_articles = sess.run(
                    self._model.temp_embedded_seq,
                    feed_dict={self._model.temp_batch: articles})
                emb_samples = sess.run(
                    self._model.temp_embedded_seq,
                    feed_dict={self._model.temp_batch: padded_outputs_ids})

                feed = {
                    discriminator.inputs: emb_samples,
                    discriminator.conditions: emb_articles,
                    discriminator.condition_lens: article_lens}
                # probs for each sample by the discriminator
                sample_dis_probs = sess.run(discriminator.dis_ypred_for_auc, feed).tolist()

                abstracts = np.tile(batch.padded_abs_ids, (len(outputs_ids), 1))
                abstract_mean_generator_probs = math.e ** self._model.run_one_batch(sess, batch, update=False, gan_eval=True)['log_gold_probs'][0]

                emb_abstracts = sess.run(
                    self._model.temp_embedded_seq,
                    feed_dict={self._model.temp_batch: abstracts})
                feed = {
                    discriminator.inputs: emb_abstracts,
                    discriminator.conditions: emb_articles,
                    discriminator.condition_lens: article_lens}
                # probs for each sample by the discriminator
                abstract_dis_probs = sess.run(discriminator.dis_ypred_for_auc, feed)[0]

                sample = randint(0, int(1 / sample_rate) if sample_rate else 0)

                sample_n = randint(0, batch_size)
                if sample == 1:
                    print()
                try:
                    decoded_words_list = outputsids2words(
                        outputs_ids, self._vocab)
                except:
                    print(outputs_ids)
                    raise

                decoded_outputs = []

                for s_n, decoded_words in enumerate(decoded_words_list):
                    try:
                        fst_stop_idx = decoded_words.index(data.STOP_DECODING)
                        decoded_words = decoded_words[:fst_stop_idx]
                    except ValueError:
                        pass
                    decoded_output = ' '.join(decoded_words)
                    if sample == 1 and s_n == sample_n:
                        print("article:\t" + original_articles[sample_n])
                        print("abstract:\t" + original_abstracts[sample_n])
                        print("hypothesis:\t" + decoded_output)
                        print("")
                    decoded_outputs.append(decoded_output)

                counter += 1  # this is how many examples we've decoded
                if counter % 10000 == 0:
                    print("Have decoded %s samples." % (counter * FLAGS.batch_size))

                if save2file:
                    for idx, sent in enumerate(original_abstracts):
                        ref_f.write(sent+"\n")
                    for idx, sent in enumerate(decoded_outputs):
                        dec_f.write(sent+"\n")
                ii = 0
                dis_rouge = []
                for output, sample_prob, sample_gen_prob in zip(
                    decoded_outputs, sample_dis_probs, sample_mean_generator_probs
                ):
                    rouge = rouge_l(original_abstracts[0].split(), output.split())
                    if ii == 0:
                        rouge_scores.append(rouge)
                        mark = 'STATS'
                    else:
                        mark = 'stats'
                    dis_rouge.append((sample_prob, rouge))
                    if save2file:
                        ove_f.write("article: "+original_articles[0]+"\n")
                        ove_f.write("reference: "+original_abstracts[0]+"\n")
                        ove_f.write("hypothesis: "+output+'\n')
                        ove_f.write("%s: --gen: %s, %s; --dis: %s, %s; --rouge: %s --\n\n" % (
                            mark, str(abstract_mean_generator_probs), str(sample_gen_prob), str(abstract_dis_probs), str(sample_prob), str(rouge)))
                        ove_f.write("\n")
                    ii += 1
                    max_dis_rouge_scores.append(sorted(dis_rouge, key=lambda x: x[1], reverse=True)[0][1])
                    dis_rouge_scores.append(sorted(dis_rouge, key=lambda x: x[0], reverse=True)[0][1])
                ove_f.write("\n")

        except KeyboardInterrupt as exc:
            print(exc)
            print("Have decoded %s samples." % (counter * FLAGS.batch_size))
            if save2file:
                ref_f.close()
                dec_f.close()
                ove_f.close()

    def write_for_discriminator(self, artcls, reference_sents, decoded_outputs):
        for artc, refe, hypo in zip(artcls, reference_sents, decoded_outputs):
            with open(os.path.join(self._hps.data_path, self._hps.mode + "_negative"), "a", 'utf-8') as f:
                f.write(hypo+"\n")
            with open(os.path.join(self._hps.data_path, self._hps.mode + "_positive"), "a", 'utf-8') as f:
                f.write(refe+"\n")
            with open(os.path.join(self._hps.data_path, self._hps.mode + "_source"), "a", 'utf-8') as f:
                f.write(artc+"\n")

    # TODO: this should be modified
    def write_for_attnvis(self, article, abstract, decoded_words, attn_dists, p_gens):
        """Write some data to json file, which can be read into the in-browser
        attention visualizer tool:
          https://github.com/abisee/attn_vis

        Args:
          article: The original article string.
          abstract: The human (correct) abstract string.
          attn_dists: List of arrays; the attention distributions.
          decoded_words: List of strings; the words of the generated summary.
          p_gens: List of scalars; the p_gen values. If not running in
          pointer-generator mode, list of None.
        """
        article_lst = article.split()  # list of words
        decoded_lst = decoded_words  # list of decoded words
        to_write = {
            'article_lst': [make_html_safe(t) for t in article_lst],
            'decoded_lst': [make_html_safe(t) for t in decoded_lst],
            'abstract_str': make_html_safe(abstract),
            'attn_dists': attn_dists
        }
        to_write['p_gens'] = p_gens
        output_fname = os.path.join(self._decode_dir, 'attn_vis_data.json')
        with open(output_fname, 'w', 'utf-8') as output_file:
            json.dump(to_write, output_file)
        tf.logging.info('Wrote visualization data to %s', output_fname)

def print_results(articles, abstracts, decoded_outputs):
    """Prints the article, the reference summmary and the decoded summary to
    screen"""
    print("")
    for article, abstract, decoded_output in zip(articles, abstracts, decoded_outputs):
        tf.logging.info('ARTICLE:  %s', article)
        tf.logging.info('REFERENCE SUMMARY: %s', abstract)
        tf.logging.info('GENERATED SUMMARY: %s', decoded_output)
        print("")


def make_html_safe(s):
    """Replace any angled brackets in string s to avoid interfering with HTML
    attention visualizer."""
    s.replace("<", "&lt;")
    s.replace(">", "&gt;")
    return s


def rouge_eval(ref_dir, dec_dir):
    """Evaluate the files in ref_dir and dec_dir with pyrouge, returning
    results_dict"""
    r = pyrouge.Rouge155()
    r.model_filename_pattern = '#ID#_reference.txt'
    r.system_filename_pattern = '(\d+)_decoded.txt'
    r.model_dir = ref_dir
    r.system_dir = dec_dir
    logging.getLogger('global').setLevel(
        logging.WARNING)  # silence pyrouge logging
    rouge_results = r.convert_and_evaluate()
    return r.output_to_dict(rouge_results)


def rouge_log(results_dict, dir_to_write):
    """Log ROUGE results to screen and write to file.

    Args:
      results_dict: the dictionary returned by pyrouge
      dir_to_write: the directory where we will write the results to"""
    log_str = ""
    for x in ["1", "2", "l"]:
        log_str += "\nROUGE-%s:\n" % x
        for y in ["f_score", "recall", "precision"]:
            key = "rouge_%s_%s" % (x, y)
            key_cb = key + "_cb"
            key_ce = key + "_ce"
            val = results_dict[key]
            val_cb = results_dict[key_cb]
            val_ce = results_dict[key_ce]
            log_str += "%s: %.4f with confidence interval (%.4f, %.4f)\n" % (
                key, val, val_cb, val_ce)
    tf.logging.info(log_str)  # log to screen
    results_file = os.path.join(dir_to_write, "ROUGE_results.txt")
    tf.logging.info("Writing final ROUGE results to %s...", results_file)
    with open(results_file, "w", 'utf-8') as f:
        f.write(log_str)


def get_decode_dir_name(hps, ckpt_name):
    """Make a descriptive name for the decode dir, including the name of the
    checkpoint we use to decode. This is called in single_pass mode."""

    dirname = hps.mode
    if FLAGS.dec_dir:
        dirname += "_%s" % FLAGS.dec_dir
    else:
        dirname += "_%s" % ckpt_name
    return dirname
