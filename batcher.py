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

"""This file contains code to process data into batches"""
from __future__ import unicode_literals, print_function
from __future__ import absolute_import
from __future__ import division

import random
import Queue
from random import shuffle
# from termcolor import colored
from threading import Thread
import time
import numpy as np
import glob
import data
import gzip
import os
import pymongo
from collections import defaultdict as dd
from codecs import open
from utils import red_assert, red_print
from data import PAD_TOKEN, POS_PAD_TOKEN, NER_PAD_TOKEN, UNKNOWN_TOKEN, START_DECODING, STOP_DECODING


def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode, "utf-8")


class Example(object):
    """Class representing a train/val/test example for text summarization."""

    def __len__(self):
        return self.enc_len

    def __init__(self, sample, enc_vocab, pos_vocab, ner_vocab, dec_vocab, hps):

        pos_tag_words = sample["pos_tag_words"]
        pos_tags = sample["pos_tags"]
        ner_tags = sample["ner_tags"]
        stem = sample["stem"]
        self.enc_tfidf = sample["tfidf_scores"]
        self.phrase_indices = sample["phrase_indices"]
        self.sent_indices = sample["sent_indices"]
        title = sample["title"]
        self.hps = hps

        # Get ids of special tokens
        start_decoding = dec_vocab.word2id(START_DECODING)
        stop_decoding = dec_vocab.word2id(STOP_DECODING)

        self.enc_len = len(pos_tag_words)
        self.enc_input = [enc_vocab.word2id(w) for w in pos_tag_words]
        self.enc_stem = [enc_vocab.word2id(w) for w in stem]
        self.enc_pos = [pos_vocab.word2id(p) for p in pos_tags]
        self.enc_ner = [ner_vocab.word2id(n) for n in ner_tags]

        # Process the abstract
        if title:
            # Get the decoder input sequence and target sequence
            self.abs_ids = [dec_vocab.word2id(w) for w in title]
            self.dec_input, self.target = self.get_dec_inp_targ_seqs(
                self.abs_ids, self.max_dec_len, start_decoding, stop_decoding)
            self.dec_len = len(self.dec_input)
        else:
            self.abs_ids, self.dec_input, self.target, self.dec_len = None

        if hps.pointer_gen:
            self.enc_input_extend_vocab, self.article_oovs = data.article2ids(pos_tag_words, self.phrase_indices, enc_vocab)
            if title:
                abs_ids_extend_vocab = data.abstract2ids(title, dec_vocab, self.article_oovs)
                _, self.target = self.get_dec_inp_targ_seqs(abs_ids_extend_vocab, hps.max_dec_steps, start_decoding, stop_decoding)

        # Store the original strings ART:
        self.original_article = ' '.join(pos_tag_words)
        self.original_abstract = title if not title else ' '.join(title)

    def get_dec_inp_targ_seqs(self, sequence, max_len, start_id, stop_id):
        inp = [start_id] + sequence[:]
        target = sequence[:]
        if len(inp) > max_len:  # truncate
            inp = inp[:max_len]
            target = target[:max_len]  # no end_token
        else:  # no truncation
            target.append(stop_id)  # end token
        red_assert(
            len(inp) == len(target),
            "abstracts and targets should be of same length but %s and %s" % (len(inp), len(target)))
        return inp, target

    def pad_decoder_inp_targ(self, max_len, pad_id):
        """Pad decoder input and target sequences with pad_id up to max_len."""
        while len(self.dec_input) < max_len:
            self.dec_input.append(pad_id)
        while len(self.target) < max_len:
            self.target.append(pad_id)

    def pad_encoder_input(self, max_len, pad_id, pos_pad_id, ner_pad_id, phrase_pad_idx, sent_pad_idx):
        """Pad the encoder input sequence with pad_id up to max_len."""
        while len(self.enc_input) < max_len:
            self.enc_input.append(pad_id)
            self.enc_stem.append(pad_id)
            self.enc_pos.append(pos_pad_id)
            self.enc_ner.append(ner_pad_id)
            self.phrase_indices.append(phrase_pad_idx)
            self.sent_indices.append(sent_pad_idx)
            self.enc_tfidf.append(0.0)


class Batch(object):
    """Class representing a minibatch of train/val/test examples for text
    summarization."""

    def __init__(self, example_list, hps, enc_vocab, pos_vocab, ner_vocab, dec_vocab):
        self.pad_id = enc_vocab.word2id(PAD_TOKEN)
        self.pos_pad_id = pos_vocab.word2id(POS_PAD_TOKEN)
        self.ner_pad_id = ner_vocab.word2id(NER_PAD_TOKEN)
        # initialize the input to the encoder
        self.init_encoder_seq(example_list, hps)
        # initialize the input and targets for the decoder
        self.init_decoder_seq(example_list, hps)
        self.store_orig_strings(example_list)  # store the original strings
        self.batch_size = len(example_list)
        # the batch size may be not the same as the hp.batch_size

    def init_encoder_seq(self, example_list, hps):
        max_enc_seq_len = max([ex.enc_len for ex in example_list])

        for ex in example_list:
            ex.pad_encoder_input(max_enc_seq_len, self.pad_id, self.pos_pad_id, self.ner_pad_id, max(self.phrase_indices)+1, max(self.sent_indices)+1)

        self.enc_batch = np.zeros((hps.batch_size, max_enc_seq_len), dtype=np.int32)
        self.stem_batch = np.zeros((hps.batch_size, max_enc_seq_len), dtype=np.int32)
        self.pos_batch = np.zeros((hps.batch_size, max_enc_seq_len), dtype=np.int32)
        self.ner_batch = np.zeros((hps.batch_size, max_enc_seq_len), dtype=np.int32)
        self.phrase_label_batch = np.zeros((hps.batch_size, max_enc_seq_len), dtype=np.int32)
        self.sent_label_batch = np.zeros((hps.batch_size, max_enc_seq_len), dtype=np.int32)
        self.tfidf_batch = np.zeros((hps.batch_size, max_enc_seq_len), dtype=np.float32)
        # self.lens_batch = np.zeros((hps.batch_size), dtype=np.int32)
        self.padding_mask_batch = np.zeros((hps.batch_size, max_enc_seq_len), dtype=np.float32)

        for i, ex in enumerate(example_list):
            self.enc_batch[i, :] = ex.enc_input[:]
            self.stem_batch[i, :] = ex.enc_stem[:]
            self.pos_batch[i, :] = ex.enc_pos[:]
            self.ner_batch[i, :] = ex.enc_ner[:]
            self.phrase_label_batch[i, :] = ex.phrase_indices[:]
            self.sent_label_batch[i, :] = ex.sent_indices[:]
            self.tfidf_batch[i, :] = ex.enc_tfidf[:]
            # self.lens_batch[i] = ex.enc_len[:]
            for j in range(ex.enc_len):
                self.padding_mask_batch[i][j] = 1

        if hps.pointer_gen:
            self.max_art_oovs = max([len(ex.article_oovs) for ex in example_list])
            self.art_oovs = [ex.article_oovs for ex in example_list]
            self.enc_batch_extend_vocab = np.zeros((hps.batch_size, max_enc_seq_len), dtype=np.int32)
            for i, ex in enumerate(example_list):
                self.enc_batch_extend_vocab[i, :] = ex.enc_input_extend_vocab[:]

    def init_decoder_seq(self, example_list, hps):
        for ex in example_list:
            if ex.abs_ids:
                ex.pad_decoder_inp_targ(hps.max_dec_steps, self.pad_id)

        self.dec_batch = np.full((hps.batch_size, hps.max_dec_steps), self.pad_id)
        self.target_batch = np.full((hps.batch_size, hps.max_dec_steps), self.pad_id)
        self.dec_padding_mask = np.zeros((hps.batch_size, hps.max_dec_steps), dtype=np.float32)
        for i, ex in enumerate(example_list):
            if ex.abs_ids:
                self.dec_batch[i, :] = ex.dec_input[:]
                self.target_batch[i, :] = ex.target[:]
                for j in range(ex.dec_len):
                    self.dec_padding_mask[i][j] = 1

    def store_orig_strings(self, example_list):
        """Store the original article and abstract strings in the Batch
        object"""
        self.original_articles = [ex.original_article for ex in example_list]  # list of lists
        self.original_abstracts = [ex.original_abstract for ex in example_list]  # list of lists
        # list of list of lists


class GenBatcher(object):

    BATCH_QUEUE_MAX = 100  # max number of batches the batch_queue can hold
    # BATCH_QUEUE_MAX = 1  # max number of batches the batch_queue can hold

    def __init__(self, file_name, mode, enc_vocab, pos_vocab, ner_vocab, dec_vocab, hps):
        self._enc_vocab = enc_vocab
        self._pos_vocab = pos_vocab
        self._ner_vocab = ner_vocab
        self._dec_vocab = dec_vocab
        self._hps = hps
        red_assert(
            mode in ["train", "test", "val"],
            "mode should be in ['train', 'test', 'val'] but %s provided" % mode)
        self._mode = mode
        self._batch_queue = Queue.Queue(self.BATCH_QUEUE_MAX)
        self._example_queue = Queue.Queue(
            self.BATCH_QUEUE_MAX * self._hps.batch_size * self._hps.beam_size)

        if mode in ["test", "val"]:
            self._num_example_q_threads = 1
            self._num_batch_q_threads = 1
            self._bucketing_cache_size = 1
            self._finished_reading = False
        else:
            self._num_example_q_threads = 16
            self._num_batch_q_threads = 4
            self._bucketing_cache_size = 100

            self._watch_thread = Thread(target=self.watch_threads)
            self._watch_thread.daemon = True
            self._watch_thread.start()

        self._example_q_threads = []
        for _ in range(self._num_example_q_threads):
            self._example_q_threads.append(
                Thread(target=self.fill_example_queue))
            self._example_q_threads[-1].daemon = True
            self._example_q_threads[-1].start()
        self._batch_q_threads = []
        for _ in range(self._num_batch_q_threads):
            self._batch_q_threads.append(Thread(target=self.fill_batch_queue))
            self._batch_q_threads[-1].daemon = True
            self._batch_q_threads[-1].start()

    def next_batch(self):
        # If the batch queue is empty, print a warning
        if self._batch_queue.qsize() == 0:
            print('Bucket input queue is empty when calling next_batch. Bucket queue size: %i, Input queue size: %i' % (
                        self._batch_queue.qsize(),
                        self._example_queue.qsize()))
            # red_assert(self._mode == "test" and self._finished_reading, "queue is empty, the mode should be 'test' but %s found" % self._mode)
            if self._mode == "test" and self._finished_reading:
                return None
            else:
                red_print("batch queue empty, waiting 2 seconds")
                time.sleep(2)

        batch = self._batch_queue.get()  # get the next Batch
        return batch

    def fill_example_queue(self):

        myclient = pymongo.MongoClient("mongodb://localhost:27017/")
        mydb = myclient["mydatabase"]
        mycol = mydb["bytecup2018"]

        try:
            if self._mode == "train":
                while True:
                    sample = mycol.aggregate([
                        {"$match": {"new_id": {"$gt": self.train_id_min, "$lt": self.train_id_max}}},
                        {"$sample": {"size": 1}}
                    ]).next()
                    example = Example(sample, self._enc_vocab, self.pos_vocab, self.ner_vocab, self._dec_vocab, self._hps)
                    self._example_queue.put(example)

            elif self._mode == "val":
                while(True):
                    samples = mycol.find(
                        {"new_id": {"$gt": self.val_id_min, "$lt": self.val_id_max}})
                    for sample in samples:
                        example = Example(sample, self._enc_vocab, self.pos_vocab, self.ner_vocab, self._dec_vocab, self._hps)
                        self._example_queue.put(example)
                    self._example_queue.put(None)

            elif self._mode == "test":
                samples = mycol.find(
                    {"new_id": {"$gt": self.test_id_min, "$lt": self.test_id_max}})
                for sample in samples:
                    example = Example(sample, self._enc_vocab, self.pos_vocab, self.ner_vocab, self._dec_vocab, self._hps)
                    self._example_queue.put(example)

        except Exception as e:
            red_print("something wrong happened while generating a sample")
            print(e)

        # "train_gan", "decode"

    def fill_batch_queue(self):
        while True:
            inputs = []
            for l in range(self._hps.batch_size * self._bucketing_cache_size):

                pair = self._example_queue.get()
                if pair:
                    if self._mode == "train" and pair not in inputs:
                        inputs.append(pair)
                    elif self._mode in ["val", 'test']:
                        inputs.append(pair)
                else:
                    inputs.append('None')
                    for _ in range(self._hps.batch_size - ((l+1) % self._hps.batch_size)):
                        inputs.append('None')
                    assert len(inputs) % self._hps.batch_size == 0, '%s, %s' % (str(len(inputs)), self._hps.batch_size)
                    break
            # sort by length of encoder sequence
            if self._mode == "train":
                inputs = sorted(inputs, key=lambda inp: len(inp))

            batches = []
            for i in range(0, len(inputs), self._hps.batch_size):
                batches.append(inputs[i:i + self._hps.batch_size])
            if self._mode == "train":
                shuffle(batches)
            for b in batches:
                if "None" in b:
                    self._batch_queue.put(None)
                    # while not testing the samples left are abandoned
                    continue
                if len(b) != self._hps.batch_size:
                    continue
                self._batch_queue.put(Batch(b, self._hps, self._enc_vocab, self._dec_vocab))
