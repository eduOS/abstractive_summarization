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

"""This file contains code to read the train/eval/test data from file and
process it, and read the vocab data from file and process it"""

from __future__ import unicode_literals, print_function
from __future__ import absolute_import
from __future__ import division

from termcolor import colored
import numpy as np
from cntk.tokenizer import text2charlist
from utils import get_phrase_oovs
import pymongo
from settings import oov_size

# <s> and </s> are used in the data files to segment the abstracts into
# sentences. They don't receive vocab ids.
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

# This has a vocab id, which is used to pad the encoder input, decoder
# input and target sequence
PAD_TOKEN = '[pad]'
# This has a vocab id, which is used to represent out-of-vocabulary words
UNKNOWN_TOKEN = 'u_n_k'
# This has a vocab id, which is used at the start of every decoder input
# sequence
START_DECODING = '[start]'
# This has a vocab id, which is used at the end of untruncated target sequences
STOP_DECODING = '[stop]'
POSITIVE_LABEL = 1
NEGATIVE_LABEL = 0

# Note: none of <s>, </s>, [PAD], [UNK], [START], [STOP] should appear in
# the vocab file.

myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["mydatabase"]
mycol = mydb["bytecup2018"]


class Vocab(object):
    """Vocabulary class for mapping between words and ids (integers)"""

    def __init__(self, _type, shared_vocab_size):
        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0
        self._shared_vocab_size = shared_vocab_size

        assert _type in ['enc_vocab', 'dec_vocab', 'stem_vocab', 'pos_vocab', 'ner_vocab']
        vocab_list = mycol.find({"_id": 'vocab_freq_dict'}).next()[_type]

        # Read the vocab file and add words up to max_size
        for word in vocab_list:
            if self._count == 0:
                assert word == PAD_TOKEN, "the first vocab should be PAD"
            if word in self._word_to_id:
                print(colored("%s already in the vocab, escape.." % word, "red"))
                continue
            self._word_to_id[word], self._id_to_word[len(self._id_to_word)] = len(self._word_to_id), word
            self._count += 1

        # print("Finished constructing vocabulary of %i total words. Last word added: %s" % (max_size, self._id_to_word[max_size-1]))
    @property
    def word_keys(self):
        return self._word_to_id.keys()

    @property
    def id_keys(self):
        return self._id_to_word.keys()

    def word2id(self, word):
        """Returns the id (integer) of a word (string). Returns [UNK] id if word
        is OOV."""
        if word not in self._word_to_id:
            return self._word_to_id[UNKNOWN_TOKEN]
        return self._word_to_id[word]

    def id2word(self, word_id):
        """Returns the word (string) corresponding to an id (integer)."""
        if word_id not in self._id_to_word:
            raise ValueError('Id not found in vocab: %d' % word_id)
        return self._id_to_word[word_id]

    @property
    def size(self):
        """Returns the total size of the vocabulary"""
        return self._count

    @property
    def shared_vocab_size(self):
        """the shared len of the enc and dec"""
        return self._shared_vocab_size


def article2ids(article_words, phrase_indices, vocab):
    ids = []
    oovs = []
    phrase_ids = []
    unk_id = vocab.word2id(UNKNOWN_TOKEN)
    phrase_oovs = get_phrase_oovs(article_words, phrase_indices)
    # article_words_phrases = list(set(article_words)) + phrase_oovs
    for w in article_words:
        i = vocab.word2id(w)
        if i == unk_id:
            if len(oovs) > oov_size:
                continue
            if w not in oovs:
                oovs.append(w)
            oov_num = oovs.index(w)
            ids.append(vocab.shared_vocab_size + oov_num)
        else:
            ids.append(i)

    # when doing attention the phrases are attended to the first layer words,
    # because order is not important in attention mechanism
    for po in phrase_oovs:
        if len(oovs) > oov_size:
            continue
        if po not in oovs:
            oovs.append(po)
        oov_num = oovs.index(po)
        phrase_ids.append(vocab.shared_vocab_size + oov_num)

    return ids, phrase_oovs, phrase_ids, oovs


def abstract2ids(abstract_words, vocab, article_oovs, title_phrase_indices):
    ids = []
    phrase_oovs = get_phrase_oovs(abstract_words, title_phrase_indices)
    shared_phrase_oovs = [po for po in phrase_oovs if po in article_oovs]

    unk_id = vocab.word2id(UNKNOWN_TOKEN)
    for w in abstract_words:
        i = vocab.word2id(w)
        if i == unk_id:
            if w in article_oovs:
                vocab_idx = vocab.size() + article_oovs.index(w)
                ids.append(vocab_idx)
            else:
                ids.append(unk_id)
        else:
            ids.append(i)
    return ids


def outputsids2words(id_ar, vocab):
    """Maps output ids to words, including mapping in-article OOVs from their
    temporary ids to the original OOV string (applicable in pointer-generator
    mode).

    Args:
      id_ar: a 2-D array of ids
      vocab: Vocabulary object

    Returns:
      words: list of words (strings)
    """
    words_lists = []
    for j, id_list in enumerate(id_ar):
        words = []
        for i in id_list:
            w = vocab.id2word(i)  # might be [UNK]
            words.append(w)
        words_lists.append(words)
    return words_lists


def show_art_oovs(articles, vocab):
    """Returns the article string, highlighting the OOVs by placing
    __underscores__ around them"""
    unk_token = vocab.word2id(UNKNOWN_TOKEN)
    out_articles = []
    for article in articles:
        words = article.split(' ')
        words = [("__%s__" % w) if vocab.word2id(
            w) == unk_token else w for w in words]
        out_str = ' '.join(words)
        out_articles.append(out_str)
    return out_articles


def show_abs_oovs(abstracts, vocab, article_oovs):
    """Returns the abstract string, highlighting the article OOVs with
    __underscores__.

    If a list of article_oovs is provided, non-article OOVs are differentiated
    like !!__this__!!.

    Args:
      abstract: string
      vocab: Vocabulary object
      article_oovs: list of list of words (strings), or None (in baseline mode)
    """
    unk_token = vocab.word2id(UNKNOWN_TOKEN)
    out_abstracts = []
    for i, abstract in enumerate(abstracts):
        words = abstract.split(' ')
        new_words = []
        for w in words:
            if vocab.word2id(w) == unk_token:  # w is oov
                if article_oovs is None:  # baseline mode
                    new_words.append("__%s__" % w)
                else:  # pointer-generator mode
                    if w in article_oovs[i]:
                        new_words.append("__%s__" % w)
                    else:
                        new_words.append("!!__%s__!!" % w)
            else:  # w is in-vocab word
                new_words.append(w)
        out_str = ' '.join(new_words)
        out_abstracts.append(out_str)
    return out_abstracts


def abstract2sents(abstract):
    """Splits abstract text from datafile into list of sentences.

    Args:
      abstract: string containing <s> and </s> tags for starts and ends of sentences

    Returns:
      sents: List of sentence strings (no tags)"""
    cur = 0
    sents = []
    while True:
        try:
            start_p = abstract.index(SENTENCE_START, cur)
            end_p = abstract.index(SENTENCE_END, start_p + 1)
            cur = end_p + len(SENTENCE_END)
            sents.append(abstract[start_p+len(SENTENCE_START):end_p])
        except ValueError:  # no more sentences
            return sents


def prepare_dis_pretraining_batch(batch):
    """
    translate the list into np array and add the targets for them
    randomly select the sample as positive or negative input
    """
    source, positives, negatives = batch
    inputs = positives + negatives

    positive_labels = [POSITIVE_LABEL for _ in positives]
    negative_labels = [NEGATIVE_LABEL for _ in negatives]
    targets = positive_labels + negative_labels

    conditions = source + source

    # randomize the inputs, conditions and targets
    assert len(inputs) == len(conditions) == len(targets)
    indices = np.random.permutation(len(inputs))
    inputs = np.array(inputs)[indices]
    conditions = np.array(conditions)[indices]
    targets = np.array(targets)[indices]

    return inputs, conditions, targets


def gen_vocab2dis_vocab(gen_ids, gen_vocab, article_oovs, dis_vocab,
                        max_len, STOP_MARK=STOP_DECODING, art_ids=None, print_sample=False):
    """
    transfer the generator vocabulary which is word based to discriminator
    vocabualry which is char based

    args:
        gen_ids: a list of list of ids (integers)
        gen_vocab: the vocabulary of the generatorj:w
        articles_oovs, if pointer_gen is true this is the temporary out of vocabulary
        dis_vocab: the vocabulary of the discriminator
        max_len: since the inputs of the discriminator can only be length fixed,
                 pads and stop_decoding all are set as 0(which is the convention in the
                 discriminator)
        STOP_MARK: the stop symbol, which is important for the sample, it may be
                   the pad symbol and etc.

    return: a two dimensional numpy array with the ids of the discriminator vocabulary
    """
    # TODO: keep the [unk] and such words
    samples_ids = []
    assert len(gen_ids) == len(article_oovs), \
        "length of gen_ids(%s) and article_oovs(%s) are not the same" % (len(gen_ids), len(article_oovs))
    samples_words = outputsids2words(gen_ids, gen_vocab, article_oovs, art_ids)
    for n, sample_words in enumerate(samples_words):
        # if print_sample:
        #     print(print_sample + ":")
        #     print(colored(" ".join(sample_words), "red"))
        try:
            fst_stop_idx = sample_words.index(STOP_MARK)  # index of the (first) [STOP] symbol
            sample_chars = text2charlist(sample_words[:fst_stop_idx], keep_word="[UNK]")
        except ValueError:
            sample_chars = text2charlist(sample_words, keep_word="[UNK]")
        if print_sample:
            print(print_sample + ":")
            print(str(n) + ": " + colored("\t".join(sample_words), "green"))
        sample_ids = [dis_vocab.word2id(char) for char in sample_chars[:max_len]]
        while len(sample_ids) < max_len:
            sample_ids.append(0)
        samples_ids.append(sample_ids)
    if print_sample:
        print('\n')
    assert len(samples_words) == len(samples_ids)
    return np.array(samples_ids)


def strip_pads(id_batch, STOP_ID):
    """
    """
    new_batch = []
    for sample_ids in id_batch:
        try:
            fst_stop_idx = sample_ids.index(STOP_ID)  # index of the (first) [STOP] symbol
            sample_ids = sample_ids[:fst_stop_idx]
        except ValueError:
            sample_ids = sample_ids
        new_batch.append(sample_ids)
    return new_batch
