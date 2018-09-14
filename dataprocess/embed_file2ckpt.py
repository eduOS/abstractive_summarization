# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
from codecs import open
from collections import defaultdict as dd
from collections import Counter
from operator import itemgetter
import random
import numpy as np
import pymongo
from ..data import PAD_TOKEN, UNKNOWN_TOKEN, START_DECODING, STOP_DECODING

myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["mydatabase"]
mycol = mydb["bytecup2018"]


def most_common(_dict, num):
        return [w for w, _ in Counter(_dict).most_common(num)]


def make_vocab(enc_vocab_size, dec_vocab_size):
    whole_vocab = list(mycol.find({"_id": 'vocab_freq_dict'}))[0]
    enc_words = whole_vocab['enc_vocab_freq_dict']
    dec_words = whole_vocab['dec_vocab_freq_dict']
    lemma_words = whole_vocab['lemma_vocab_freq_dict']

    shared_part = most_common(dict(dec_words + [[PAD_TOKEN, float('inf')], [UNKNOWN_TOKEN, 10**6]]), dec_vocab_size-2)
    dec_part = [START_DECODING, STOP_DECODING]
    dec_vocab = shared_part + dec_part
    assert len(dec_vocab) == dec_vocab_size, 'dec_vocab should be of length %s but %s' % (dec_vocab_size, len(dec_vocab))

    enc_words = dict(enc_words + lemma_words)
    for s in shared_part:
        if s in enc_words:
            del enc_words[s]

    enc_part = most_common(enc_words, enc_vocab_size - len(shared_part))
    enc_vocab = shared_part + enc_part
    assert len(enc_vocab) == enc_vocab_size, 'enc_vocab should be of length %s but %s' % (enc_vocab_size, len(enc_vocab))
    mycol.update_many(
        {"_id": "vocab_freq_dict"},
        {"$set": {
            "shared_vocab_size": dec_vocab_size-2,
            "enc_vocab_size": enc_vocab_size,
            "dec_vocab_size": dec_vocab_size,
            "enc_vocab": enc_vocab,
            "dec_vocab": dec_vocab,
        }
        }, upsert=False)
    return shared_part, enc_part, dec_part


def check_emb(embed_path, enc_vocab_size, dec_vocab_size):
    shared_part, enc_part, dec_part = make_vocab(enc_vocab_size, dec_vocab_size)
    glove_embed_dic = {}
    max_ = 0
    min_ = 0
    embed_f = open(embed_path, 'r', 'utf-8')
    for embed in embed_f:
        try:
            s_embed = embed.split()
            w, emb = s_embed[:1].strp(), list(map(float, s_embed[1:]))
        except:
            print(embed)
        if max(emb) > max_:
            max_ = max(emb)
        if min(emb) < min_:
            min_ = min(emb)

        # ensure the equal length
        assert 300 == len(emb)
        glove_embed_dic[w] = emb

    def df():
        return [round(random.uniform(min_, max_), 6) for i in range(300)]

    doed = dd(df, glove_embed_dic)

    shared_emb = itemgetter(*shared_part)(doed)
    enc_emb = itemgetter(*enc_part)(doed)
    dec_emb = itemgetter(*dec_part)(doed)

    return shared_emb, enc_emb, dec_emb

embed_path, enc_vocab_size, dec_vocab_size = "./data/glove.txt", 7500, 7500
shared_emb, enc_emb, dec_emb = check_emb(embed_path, enc_vocab_size, dec_vocab_size)

np.save('./data/shared_part_embeddings.npy', np.array(shared_emb))
np.save('./data/enc_part_embeddings.npy', np.array(enc_emb))
np.save('./data/dec_part_embeddings.npy', np.array(dec_emb))

# shared_part_emb_len = len(shared_emb)
# enc_part_emb_len = len(enc_emb)
# dec_part_emb_len = len(dec_emb)

# shared_part_emb_ph = tf.placeholder(
#     tf.float32, [shared_part_emb_len, 300], name='shared_embeddings')
# enc_part_emb_ph = tf.placeholder(
#     tf.float32, [enc_part_emb_len, 300], name='enc_embeddings')
# dec_part_emb_ph = tf.placeholder(
#     tf.float32, [dec_part_emb_len, 300], name='dec_embeddings')

# shared_part_embeddings = tf.get_variable(
#     "shared_part_embeddings", [shared_part_emb_len, 300], dtype=tf.float32)
# enc_part_embeddings = tf.get_variable(
#     "enc_part_embeddings", [enc_part_emb_len, 300], dtype=tf.float32)
# dec_part_embeddings = tf.get_variable(
#     "dec_part_embeddings", [dec_part_emb_len, 300], dtype=tf.float32)
# saver = tf.train.Saver({
#     "shared_part_embeddings": shared_part_embeddings,
#     "enc_part_embeddings": enc_part_embeddings,
#     "dec_part_embeddings": dec_part_embeddings})

# shared_as_op = shared_part_embeddings.assign(shared_part_emb_ph)
# enc_as_op = enc_part_embeddings.assign(enc_part_emb_ph)
# dec_as_op = dec_part_embeddings.assign(dec_part_emb_ph)

# sess = tf.Session()
# sess.run(shared_as_op, feed_dict={shared_part_emb_ph: shared_emb})
# sess.run(enc_as_op, feed_dict={enc_part_emb_ph: enc_emb})
# sess.run(dec_as_op, feed_dict={dec_part_emb_ph: dec_emb})
# save_path = sys.argv[-1]  # this should be file path not directory path

# saver.save(sess, save_path)
