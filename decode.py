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
import time
import tensorflow as tf
import beam_search
import data
import json
from codecs import open
# import pyrouge
import gen_utils
import logging
from six.moves import xrange
# import numpy as np

SECS_UNTIL_NEW_CKPT = 60  # max number of seconds before loading new checkpoint


class BeamSearchDecoder(object):
    """Beam search decoder."""

    def __init__(self, saver, sess, model, vocab):
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

        # Load an initial checkpoint to use for decoding
        ckpt_path = gen_utils.load_ckpt(saver, self._sess)
        print(ckpt_path)
        # the checkpoint should add timestamp

        if self._hps.single_pass:
            # Make a descriptive decode directory name
            # this is something of the form "ckpt-123456"
            # ckpt_name = "ckpt-" + ckpt_path.split('-')[-1]
            ckpt_name = str(time.time())[:10]
            self._decode_dir = os.path.join(
                self._hps.log_root, get_decode_dir_name(self._hps, ckpt_name))
            if os.path.exists(self._decode_dir):
                raise Exception(
                    "single_pass decode directory %s should not already exist" %
                    self._decode_dir)

        else:  # Generic decode dir name
            self._decode_dir = os.path.join(self._hps.log_root, "decode")

        # Make the decode dir if necessary
        if not os.path.exists(self._decode_dir):
            os.mkdir(self._decode_dir)

        if self._hps.single_pass:
            # Make the dirs to contain output written in the correct format for
            # pyrouge
            self._rouge_ref_dir = os.path.join(self._decode_dir, "reference")
            if not os.path.exists(self._rouge_ref_dir):
                os.mkdir(self._rouge_ref_dir)
            self._rouge_dec_dir = os.path.join(self._decode_dir, "decoded")
            if not os.path.exists(self._rouge_dec_dir):
                os.mkdir(self._rouge_dec_dir)

    def generate(self, batcher, include_start_token=False):
        # the abstract should also be generated
        batch = batcher.next_batch()
        if batch is None:
            return

        # Run beam search to get best Hypothesis
        enc_states, dec_in_state, best_hyps = beam_search.run_beam_search(self._sess, self._model, self._vocab, batch)

        # Extract the output ids from the hypothesis and convert back to
        # words
        if include_start_token:
            outputs_ids = [[int(t) for t in best_hyp.tokens[:]] for best_hyp in best_hyps]
        else:
            outputs_ids = [[int(t) for t in best_hyp.tokens[1:]] for best_hyp in best_hyps]
        return batch, enc_states, dec_in_state, outputs_ids

    def decode(self, batcher):
        """Decode examples until data is exhausted (if self._hps.single_pass) and
        return, or decode indefinitely, loading latest checkpoint at regular
        intervals"""
        t0 = time.time()
        counter = 0
        while True:
            batch = batcher.next_batch()
            # 1 example repeated across batch
            if batch is None:
                # finished decoding dataset in single_pass mode
                assert self._hps.single_pass, (
                    "Dataset exhausted, but we are not in single_pass mode")
                tf.logging.info(
                    "Decoder has finished reading dataset for single_pass.")
                tf.logging.info(
                    "Output has been saved in %s and %s. \
                    Now starting ROUGE eval...",
                    self._rouge_ref_dir,
                    self._rouge_dec_dir)
                # results_dict = rouge_eval(
                #     self._rouge_ref_dir, self._rouge_dec_dir)
                # rouge_log(results_dict, self._decode_dir)
                return

            original_articles = batch.original_articles
            original_abstracts = batch.original_abstracts
            # original_abstract_sents = batch.original_abstracts_sents[0]
            # list of strings

            art_oovs = [batch.art_oovs[i]
                        for i in xrange(self._hps.batch_size)]
            articles_withunks = data.show_art_oovs(original_articles, self._vocab)
            abstracts_withunks = data.show_abs_oovs(original_abstracts, self._vocab,
                                                    (art_oovs if self._hps.pointer_gen else None))

            # Run beam search to get best Hypothesis
            _, _, best_hyps = beam_search.run_beam_search(self._sess, self._model, self._vocab, batch)
            # is the beam_size here 1?
            outputs_ids = [[int(t) for t in hyp.tokens[1:]] for hyp in best_hyps]
            for output_ids in outputs_ids:
                print(output_ids)
                time.sleep(2)

            decoded_words_list = data.outputsids2words(
                outputs_ids, self._vocab, (art_oovs if self._hps.pointer_gen else None))
            # art_oovs[0] should be changed, batch size examples should be
            # concluded
            decoded_outputs = []

            # Remove the [STOP] token from decoded_words, if necessary
            for decoded_words in decoded_words_list:
                try:
                    fst_stop_idx = decoded_words.index(data.STOP_DECODING)
                    decoded_words = decoded_words[:fst_stop_idx]
                except ValueError:
                    pass
                decoded_outputs.append(' '.join(decoded_words))

            if self._hps.single_pass:
                # write ref summary and decoded summary to file, to eval with
                # pyrouge later
                self.write_for_rouge(
                    original_articles, original_abstracts, decoded_outputs, counter)
                self.write_for_discriminator(
                    original_articles, original_abstracts, decoded_outputs)
                counter += 1  # this is how many examples we've decoded
            else:
                print_results(articles_withunks, abstracts_withunks, decoded_outputs)
                # log output to screen
                self.write_for_attnvis(articles_withunks, abstracts_withunks,
                                       decoded_words, best_hyps.attn_dists, best_hyps.p_gens)
                # write info to .json file for visualization tool

                # Check if SECS_UNTIL_NEW_CKPT has elapsed; if so return so we
                # can load a new checkpoint
                t1 = time.time()
                if t1-t0 > SECS_UNTIL_NEW_CKPT:
                    tf.logging.info(
                        'We\'ve been decoding with same checkpoint for %i \
                        seconds. Time to load new checkpoint',
                        t1-t0)
                    _ = gen_utils.load_ckpt(self._saver, self._sess) # NOQA
                    t0 = time.time()

    def write_for_discriminator(self, artcls, reference_sents, decoded_outputs):
        for artc, refe, hypo in zip(artcls, reference_sents, decoded_outputs):
            with open(os.path.join(self._hps.data_path, self._hps.mode + "_negative"), "a", 'utf-8') as f:
                f.write(hypo+"\n")
            with open(os.path.join(self._hps.data_path, self._hps.mode + "_positive"), "a", 'utf-8') as f:
                f.write(refe+"\n")
            with open(os.path.join(self._hps.data_path, self._hps.mode + "_source"), "a", 'utf-8') as f:
                f.write(artc+"\n")

    def write_for_rouge(self, artcls, original_abstracts, decoded_outputs, ex_index):
        """Write output to file in correct format for eval with pyrouge. This is
        called in single_pass mode.

        Args:
          decoded_words: list of strings
          ex_index: int, the index with which to label the files
        """
        # First, divide decoded output into sentences

        # pyrouge calls a perl script that puts the data into HTML files.
        # Therefore we need to make our output HTML safe.
        decoded_sents = [make_html_safe(w) for w in decoded_outputs]
        reference_sents = [make_html_safe(w) for w in original_abstracts]
        artcls = [make_html_safe(w) for w in artcls]

        # Write to file
        ref_file = os.path.join(
            self._rouge_ref_dir,
            "reference.txt")
        decoded_file = os.path.join(
            self._rouge_dec_dir,
            "decoded.txt")
        overview_file = os.path.join(
            self._decode_dir,
            "overview.txt")

        with open(ref_file, "a", 'utf-8') as f:
            for idx, sent in enumerate(reference_sents):
                f.write(sent+"\n")
        with open(decoded_file, "a", 'utf-8') as f:
            for idx, sent in enumerate(decoded_sents):
                print(sent)
                f.write(sent+"\n")
        with open(overview_file, "a", 'utf-8') as f:
            for artc, refe, hypo in zip(artcls, reference_sents, decoded_sents):
                f.write("article: "+artc+"\n")
                f.write("reference: "+refe+"\n")
                f.write("hypothesis: "+hypo+"\n")
                f.write("\n")

        tf.logging.info("Wrote example %i to file" % ex_index)

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
        if self._hps.pointer_gen:
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
    if ckpt_name is not None:
        dirname += "_%s" % ckpt_name
    return dirname
