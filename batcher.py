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
from threading import Thread
import time
import numpy as np
import tensorflow as tf
import glob
import data
import gzip
import os
from cntk.tokenizer import text2charlist
from codecs import open
import datetime


def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode, "utf-8")


class Example(object):
    """Class representing a train/val/test example for text summarization."""

    def __init__(self, article, abstract, vocab, hps):
        """Initializes the Example, performing tokenization and truncation to
        produce the encoder, decoder and target sequences, which are stored in
        self.

        Args:
          article: source text; a string. each token is separated by a single
          space.
          vocab: Vocabulary object
          hps: hyperparameters
        """
        self.hps = hps

        # Get ids of special tokens
        start_decoding = vocab.word2id(data.START_DECODING)
        stop_decoding = vocab.word2id(data.STOP_DECODING)

        # Process the article
        article_words = article.split()
        if len(article_words) > hps.max_enc_steps:
            article_words = article_words[:hps.max_enc_steps]
        # store the length after truncation but before padding
        self.enc_len = len(article_words)
        # list of word ids; OOVs are represented by the id for UNK token
        self.enc_input = [vocab.word2id(w) for w in article_words]

        # Process the abstract
        abstract_words = abstract.split()  # list of strings
        # list of word ids; OOVs are represented by the id for UNK token
        self.abs_ids = [vocab.word2id(w) for w in abstract_words]

        # Get the decoder input sequence and target sequence
        self.dec_input, self.target = self.get_dec_inp_targ_seqs(
            self.abs_ids, hps.max_dec_steps, start_decoding, stop_decoding)
        self.dec_len = len(self.dec_input)

        # If using pointer-generator mode, we need to store some extra info
        if hps.pointer_gen:
            # Store a version of the enc_input where in-article OOVs are
            # represented by their temporary OOV id; also store the in-article
            # OOVs words themselves
            self.enc_input_extend_vocab, self.article_oovs = \
                data.article2ids(article_words, vocab)

            # Get a verison of the reference summary where in-article OOVs are
            # represented by their temporary article OOV id
            self.abs_ids_extend_vocab = data.abstract2ids(
                abstract_words, vocab, self.article_oovs)

            # Overwrite decoder target sequence so it uses the temp article OOV
            # ids
            _, self.target = self.get_dec_inp_targ_seqs(
                self.abs_ids_extend_vocab,
                hps.max_dec_steps,
                start_decoding,
                stop_decoding)

        # Store the original strings ART:
        self.original_article = article[5:]
        self.original_abstract = abstract[5:]
        # self.original_abstract_sents = abstract_sentences

    def get_dec_inp_targ_seqs(self, sequence, max_len, start_id, stop_id):
        """Given the reference summary as a sequence of tokens, return the input
        sequence for the decoder, and the target sequence which we will use to
        calculate loss. The sequence will be truncated if it is longer than
        max_len. The input sequence must start with the start_id and the target
        sequence must end with the stop_id (but not if it's been truncated).

        Args:
          sequence: List of ids (integers)
          max_len: integer
          start_id: integer
          stop_id: integer

        Returns:
          inp: sequence length <=max_len starting with start_id
          target: sequence same length as input, ending with stop_id only if
          there was no truncation
        """
        inp = [start_id] + sequence[:]
        target = sequence[:]
        if len(inp) > max_len:  # truncate
            inp = inp[:max_len]
            target = target[:max_len]  # no end_token
        else:  # no truncation
            target.append(stop_id)  # end token
        assert len(inp) == len(target)
        return inp, target

    def pad_decoder_inp_targ(self, max_len, pad_id):
        """Pad decoder input and target sequences with pad_id up to max_len."""
        while len(self.dec_input) < max_len:
            self.dec_input.append(pad_id)
        while len(self.target) < max_len:
            self.target.append(pad_id)

    def pad_encoder_input(self, max_len, pad_id):
        """Pad the encoder input sequence with pad_id up to max_len."""
        while len(self.enc_input) < max_len:
            self.enc_input.append(pad_id)
        if self.hps.pointer_gen:
            while len(self.enc_input_extend_vocab) < max_len:
                self.enc_input_extend_vocab.append(pad_id)

    def pad_abs(self, max_len, pad_id):
        """Pad the encoder input sequence with pad_id up to max_len."""
        while len(self.abs_ids) < max_len:
            self.abs_ids.append(pad_id)
        if self.hps.pointer_gen:
            while len(self.abs_ids_extend_vocab) < max_len:
                self.abs_ids_extend_vocab.append(pad_id)


class Batch(object):
    """Class representing a minibatch of train/val/test examples for text
    summarization."""

    def __init__(self, example_list, hps, vocab):
        """Turns the example_list into a Batch object.

        Args:
           example_list: List of Example objects
           hps: hyperparameters
           vocab: Vocabulary object
        """
        self.pad_id = vocab.word2id(
            data.PAD_TOKEN)  # id of the PAD token used to pad sequences
        # initialize the input to the encoder
        self.init_encoder_seq(example_list, hps)
        # initialize the input and targets for the decoder
        self.init_decoder_seq(example_list, hps)
        self.store_orig_strings(example_list)  # store the original strings
        self.batch_size = len(example_list)
        # the batch size may be not the same as the hp.batch_size

    def init_encoder_seq(self, example_list, hps):
        """Initializes the following:
            self.enc_batch:
              numpy array of shape (batch_size, <=max_enc_steps) containing
              integer ids (all OOVs represented by UNK id), padded to length of
              longest sequence in the batch
            self.enc_lens:
              numpy array of shape (batch_size) containing integers. The
              (truncated) length of each encoder input sequence (pre-padding).

          If hps.pointer_gen, additionally initializes the following:
            self.max_art_oovs:
              maximum number of in-article OOVs in the batch
            self.art_oovs:
              list of list of in-article OOVs (strings), for each example in the
              batch
            self.enc_batch_extend_vocab:
              Same as self.enc_batch, but in-article OOVs are represented by
              their temporary article OOV number.
        """
        # Determine the maximum length of the encoder input sequence in this
        # batch
        max_enc_seq_len = max([ex.enc_len for ex in example_list])
        # the  length for each batch is different

        # Pad the encoder input sequences up to the length of the longest
        # sequence
        for ex in example_list:
            ex.pad_encoder_input(max_enc_seq_len, self.pad_id)

        for ex in example_list:
            ex.pad_abs(hps.max_enc_steps, self.pad_id)

        # Initialize the numpy arrays
        # Note: our enc_batch can have different length (second dimension) for
        # each batch because we use dynamic_rnn for the encoder.
        self.enc_batch = np.zeros((hps.batch_size, max_enc_seq_len), dtype=np.int32)
        self.enc_lens = np.zeros((hps.batch_size), dtype=np.int32)
        self.enc_padding_mask = np.zeros((hps.batch_size, max_enc_seq_len), dtype=np.float32)

        # Fill in the numpy arrays
        for i, ex in enumerate(example_list):
            self.enc_batch[i, :] = ex.enc_input[:]
            self.enc_lens[i] = ex.enc_len
            for j in range(ex.enc_len):
                self.enc_padding_mask[i][j] = 1

        # For pointer-generator mode, need to store some extra info
        if hps.pointer_gen:
            # Determine the max number of in-article OOVs in this batch
            self.max_art_oovs = max([len(ex.article_oovs) for ex in example_list])
            # Store the in-article OOVs themselves
            self.art_oovs = [ex.article_oovs for ex in example_list]
            # Store the version of the enc_batch that uses the article OOV ids
            self.enc_batch_extend_vocab = np.zeros((hps.batch_size, max_enc_seq_len), dtype=np.int32)
            for i, ex in enumerate(example_list):
                self.enc_batch_extend_vocab[i, :] = ex.enc_input_extend_vocab[:]

    def init_decoder_seq(self, example_list, hps):
        """Initializes the following:
            self.dec_batch:
              numpy array of shape (batch_size, max_dec_steps), containing
              integer ids as input for the decoder, padded to max_dec_steps
              length.
            self.target_batch:
              numpy array of shape (batch_size, max_dec_steps), containing
              integer ids for the target sequence, padded to max_dec_steps
              length.
            self.padding_mask:
              numpy array of shape (batch_size, max_dec_steps), containing 1s
              and 0s. 1s correspond to real tokens in dec_batch and
              target_batch; 0s correspond to padding.
            """
        # Pad the inputs and targets
        for ex in example_list:
            ex.pad_decoder_inp_targ(hps.max_dec_steps, self.pad_id)

        # Initialize the numpy arrays.
        # Note: our decoder inputs and targets must be the same length for each
        # batch (second dimension = max_dec_steps) because we do not use a
        # dynamic_rnn for decoding. However I believe this is possible, or will
        # soon be possible, with Tensorflow 1.0, in which case it may be best to
        # upgrade to that.
        self.dec_batch = np.zeros((hps.batch_size, hps.max_dec_steps), dtype=np.int32)
        self.target_batch = np.zeros((hps.batch_size, hps.max_dec_steps), dtype=np.int32)
        self.padding_mask = np.zeros((hps.batch_size, hps.max_dec_steps), dtype=np.float32)

        # Fill in the numpy arrays
        for i, ex in enumerate(example_list):
            self.dec_batch[i, :] = ex.dec_input[:]
            self.target_batch[i, :] = ex.target[:]
            for j in range(ex.dec_len):
                self.padding_mask[i][j] = 1

    def store_orig_strings(self, example_list):
        """Store the original article and abstract strings in the Batch
        object"""
        self.original_articles = [ex.original_article for ex in example_list]  # list of lists
        self.original_abstracts = [ex.original_abstract for ex in example_list]  # list of lists
        # list of list of lists


class GenBatcher(object):
    """A class to generate minibatches of data. Buckets examples together based
    on length of the encoder sequence."""
    # TODO: bucket can be added

    BATCH_QUEUE_MAX = 100  # max number of batches the batch_queue can hold

    def __init__(self, mode, vocab, hps, single_pass):
        """Initialize the batcher. Start threads that process the data into
        batches.

        Args:
          data_path: tf.Example filepattern.
          vocab: Vocabulary object
          hps: hyperparameters
          single_pass: If True, run through the dataset exactly once (useful for
          when you want to run evaluation on the dev or test set). Otherwise
          generate random batches indefinitely (useful for training).
        """
        self._vocab = vocab
        self._hps = hps
        self._mode = mode
        self._single_pass = single_pass
        self._data_path = os.path.join(hps.data_path, mode) + "*"
        if self._mode == "val":
            self._single_pass = True

        # Initialize a queue of Batches waiting to be used, and a queue of
        # Examples waiting to be batched
        self._batch_queue = Queue.Queue(self.BATCH_QUEUE_MAX)
        self._example_queue = Queue.Queue(
            self.BATCH_QUEUE_MAX * self._hps.batch_size * self._hps.beam_size)

        # Different settings depending on whether we're in single_pass mode or
        # not
        if single_pass:
            # just one thread, so we read through the dataset just once
            self._num_example_q_threads = 1
            self._num_batch_q_threads = 1  # just one thread to batch examples
            # only load one batch's worth of examples before bucketing; this
            # essentially means no bucketing
            self._bucketing_cache_size = 1
            # this will tell us when we're finished reading the dataset
            self._finished_reading = False
        else:
            self._num_example_q_threads = 16
            # num threads to fill example queue
            self._num_batch_q_threads = 4  # num threads to fill batch queue
            # how many batches-worth of examples to load into cache before
            # bucketing
            self._bucketing_cache_size = 100

        # Start the threads that load the queues
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

        # Start a thread that watches the other threads and restarts them if
        # they're dead
        if not single_pass:
            # We don't want a watcher in single_pass mode because the threads
            # shouldn't run forever
            self._watch_thread = Thread(target=self.watch_threads)
            self._watch_thread.daemon = True
            self._watch_thread.start()

    def next_batch(self):
        """Return a Batch from the batch queue.

        If mode='decode' then each batch contains a single example repeated
        beam_size-many times; this is necessary for beam search.

        Returns:
          batch: a Batch object, or None if we're in single_pass mode and we've
          exhausted the dataset.
        """
        # If the batch queue is empty, print a warning
        if self._batch_queue.qsize() == 0:
            print('Bucket input queue is empty when calling next_batch. Bucket \
                    queue size: %i, Input queue size: %i' % (
                        self._batch_queue.qsize(),
                        self._example_queue.qsize()))
            if self._single_pass and self._finished_reading:
                tf.logging.info("Finished reading dataset in single_pass mode.")
                return None

        batch = self._batch_queue.get()  # get the next Batch
        return batch

    def fill_example_queue(self):
        """Reads data from file and processes into Examples which are then
        placed into the example queue."""

        input_gen = self.text_generator()

        while True:
            try:
                # read the next example from file. article and abstract are both
                # strings.
                (article, abstract) = input_gen.next()
            except StopIteration:  # if there are no more examples:
                tf.logging.info(
                    "The example generator for this example queue filling thread\
                    has exhausted data.")
                if self._single_pass:
                    tf.logging.info(
                        "single_pass mode is on, so we've finished reading\
                        dataset. This thread is stopping.")
                    self._finished_reading = True
                    break
                else:
                    raise Exception(
                        "single_pass mode is off but the example generator is\
                        out of data; error.")

            # abstract_sentences = [
            #     abstract
                # sent.strip() for sent in data.abstract2sents(abstract)
            # ]
            # Use the <s> and </s> tags in abstract to get a list of sentences.
            # Process into an Example.
            if article and abstract:
                example = Example(
                    article, abstract, self._vocab, self._hps)
                # what is the vocab here? the extended vocab?
                # place the Example in the example queue.
                self._example_queue.put(example)
            else:
                self._example_queue.put(None)

    def fill_batch_queue(self):
        """Takes Examples out of example queue, sorts them by encoder sequence
        length, processes into Batches and places them in the batch queue.

        In decode mode, makes batches that each contain a single example
        repeated.
        """
        while True:
            # Get bucketing_cache_size-many batches of Examples into a list,
            # then sort
            inputs = []
            for _ in range(self._hps.batch_size * self._bucketing_cache_size):
                pair = self._example_queue.get()
                if pair:
                    inputs.append(pair)
                else:
                    self._batch_queue.put(None)
                    continue
            # sort by length of encoder sequence
            inputs = sorted(inputs, key=lambda inp: inp.enc_len)

            # Group the sorted Examples into batches, optionally shuffle the
            # batches, and place in the batch queue.
            batches = []
            for i in range(0, len(inputs), self._hps.batch_size):
                batches.append(inputs[i:i + self._hps.batch_size])
            if not self._single_pass:
                shuffle(batches)
            for b in batches:  # each b is a list of Example objects
                self._batch_queue.put(Batch(b, self._hps, self._vocab))

    def watch_threads(self):
        """Watch example queue and batch queue threads and restart if dead."""
        while True:
            time.sleep(60)
            for idx, t in enumerate(self._example_q_threads):
                if not t.is_alive():  # if the thread is dead
                    print('Found example queue thread dead. Restarting.')
                    new_t = Thread(target=self.fill_example_queue)
                    self._example_q_threads[idx] = new_t
                    new_t.daemon = True
                    new_t.start()
            for idx, t in enumerate(self._batch_q_threads):
                if not t.is_alive():  # if the thread is dead
                    print('Found batch queue thread dead. Restarting.')
                    new_t = Thread(target=self.fill_batch_queue)
                    self._batch_q_threads[idx] = new_t
                    new_t.daemon = True
                    new_t.start()

    def text_generator(self):
        """read abstract and article pairs directly from file

        Args:
            data_dir: where to find the text files
            single_pass: if the single pass
        """
        while True:
            filelist = glob.glob(self._data_path)  # get the list of datafiles
            if self._mode == "val":
                assert len(filelist) == 1, \
                    "in val mode the len should be 1 but %s given." % len(filelist)
            assert filelist, ('Error: Empty filelist at %s' % self._data_path)
            if self._single_pass:
                filelist = sorted(filelist)
            else:
                random.shuffle(filelist)
            for ff in filelist:
                print("opening file %s" % ff)
                f = open(ff, "r", 'utf-8')
                while True:
                    art_abs = f.readline().strip().split("\t")
                    if len(art_abs) != 2:
                        print(
                            "file %s reaches the end of the data file %s"
                            % (f.name, datetime.datetime.now().strftime("on %m-%d at %H:%M")))
                        if self._mode == "val":
                            f.seek(0)
                            yield (None, None)
                            continue
                        else:
                            f.close()
                            print("closing file %s" % ff)
                            break
                    article_text, abstract_text = art_abs
                    if article_text and abstract_text:
                        yield (article_text, abstract_text)
                    else:
                        print('Found an example with empty article text. Skipping it.')


def get_batch(self, data, batch_size, balance=False, put_back=True):
  encoder_inputs = []
  targets = []

  # Get a random batch of encoder and decoder inputs from data,
  # pad them if needed, reverse encoder inputs and add GO to decoder.
  for _ in range(batch_size):

    if len(data) == 0:
      break

    else:
      if put_back and not balance:
        encoder_input, target = random.choice(data)
      if put_back and balance:
        clas = random.choice(data)
        encoder_input, target = random.choice(clas)
      else:
        encoder_input, target = data.pop()
      # add to the batch
      encoder_inputs.append(encoder_input)
      targets.append(target)

  if len(targets) == 0:
    return None, None, False
  else:
    encoder_inputs = list(np.transpose(np.array(encoder_inputs)))
    targets = np.array(targets)
    return encoder_inputs, targets, True


class DisBatcher:
    """
    all training data have a compared negative abstract which can be ignored in the gan training
    """

    def __init__(self, data_dir, mode, gen_vocab, dis_vocab, batch_size=1, max_art_steps=80, max_abs_steps=15, single_pass=False, clip_length=True):
        self.positive = fopen(os.path.join(data_dir, mode + "_positive"), 'r')
        self.negative = fopen(os.path.join(data_dir, mode + "_negative"), 'r')
        self.source = fopen(os.path.join(data_dir, mode + "_source"), 'r')
        self.dis_vocab = dis_vocab
        self.gen_vocab_keys = gen_vocab.word_keys

        self.batch_size = batch_size
        self.max_art_steps = max_art_steps
        self.max_abs_steps = max_abs_steps
        self.end_of_data = False
        self.single_pass = single_pass
        self.clip_length = clip_length

    def __iter__(self):
        return self

    def reset(self):
        self.positive.seek(0)
        self.negative.seek(0)
        self.source.seek(0)

    def next_batch(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            # raise StopIteration

        positive = []
        negative = []
        source = []

        try:
            while True:
                abs_p = self.positive.readline().strip()
                abs_n = self.negative.readline().strip()
                art = self.source.readline().strip()

                if abs_p == "" or art == "":
                    raise IOError
                if abs_n == "":
                    # the generated negative abstract may be empty
                    continue
                gen_vocab = art.split() + self.gen_vocab_keys
                abs_p = abs_p.split()
                abs_p = ' '.join([p if p in gen_vocab else "[UNK]" for p in abs_p])

                abs_p = text2charlist(abs_p, keep_word='[UNK]')
                abs_n = text2charlist(abs_n, keep_word='[UNK]')
                art = text2charlist(art, keep_word='[UNK]')

                if not self.clip_length:
                    if len(abs_p) > self.max_abs_steps or len(abs_n) > self.max_abs_steps or len(art) > self.max_art_steps:
                        continue
                else:
                    abs_p = abs_p[:self.max_abs_steps]
                    abs_n = abs_n[:self.max_abs_steps]
                    art = art[:self.max_art_steps]

                abs_p = [self.dis_vocab.word2id(w) for w in abs_p]
                abs_n = [self.dis_vocab.word2id(w) for w in abs_n]
                art = [self.dis_vocab.word2id(w) for w in art]

                abs_p = abs_p + [0] * (self.max_abs_steps - len(abs_p))
                abs_n = abs_n + [0] * (self.max_abs_steps - len(abs_n))
                art = art + [0] * (self.max_art_steps - len(art))

                positive.append(abs_p)
                negative.append(abs_n)
                source.append(art)

                if len(positive) >= self.batch_size:
                    break
        except IOError:
            if self.single_pass:
                return None, None, None
            else:
                print("Reaches the end..")
            self.end_of_data = True

        if len(positive) <= 0 or len(negative) <= 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        return source, positive, negative
