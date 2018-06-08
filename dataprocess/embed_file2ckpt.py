# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
from codecs import open
from collections import defaultdict as dd
import random
import sys
import os

# generate checkpoint from vocab and embeddings
# python embed_file2ckpt.py vocab_file embed_file vocab_size out_dir
# python dataprocess/embed_file2ckpt.py data/vocab ../../data/zh_emb/emb_wd/embedding.300 100000 ./temp/
finished_files_dir = "./temp/"
if not os.path.exists(finished_files_dir):
    os.makedirs(finished_files_dir)
finished_files_dir = "./temp/"


def read_from_file(vocab_path, embed_path, vocab_size):
    count = 0
    emb_dim = 0
    old_embed_dic = {}
    new_embed_l = []
    max_ = 0
    min_ = 0
    embed_f = open(embed_path, 'r', 'utf-8')
    for embed in embed_f:
        try:
            v, emb = embed.strip().split("\t")
        except:
            print(embed)
        v = v.strip()
        if v in old_embed_dic:
            continue
        emb_l = [float(e) for e in emb.split()]
        if max(emb_l) > max_:
            max_ = max(emb_l)
        if min(emb_l) < min_:
            min_ = min(emb_l)

        if emb_dim == 0:
            emb_dim = len(emb_l)
        else:
            assert emb_dim == len(emb_l)
        old_embed_dic[v] = emb_l

    def df():
        return [round(random.uniform(min_, max_), 6) for i in range(emb_dim)]

    doed = dd(df, old_embed_dic)

    vocab_f = open(vocab_path, 'r', 'utf-8')
    v_l = []
    while(count < vocab_size):
        v, c, _ = vocab_f.readline().strip().split()
        v = v.strip()
        if v in v_l:
            raise "%s is seen previously" % v
        new_embed_l.append(doed[v])
        v_l.append(v)
        count += 1
        if count % 2000 == 0:
            print(count)

    return new_embed_l, emb_dim

enc_vocab_path, enc_embed_path, enc_vocab_size = "./data/enc_vocab", "../../data/zh_emb/emb_wd/embedding.300", 500000
dec_vocab_path, dec_embed_path, dec_vocab_size = "./data/dec_vocab", "../../data/zh_emb/emb_ch/embedding.300", 7500
enc_emb_l, enc_emb_dim = read_from_file(enc_vocab_path, enc_embed_path, enc_vocab_size)
dec_emb_l, dec_emb_dim = read_from_file(dec_vocab_path, dec_embed_path, dec_vocab_size)
# vocab_size = 100000
# emb_dim = 300
# emb_l = [[1.0] * emb_dim] * vocab_size

assert enc_emb_dim == dec_emb_dim
emb_dim = enc_emb_dim

enc_emb_ph = tf.placeholder(
    tf.float32, [enc_vocab_size, emb_dim], name='enc_embeddings')
dec_emb_ph = tf.placeholder(
    tf.float32, [dec_vocab_size, emb_dim], name='dec_embeddings')
enc_embeddings = tf.get_variable(
    "enc_embeddings", [enc_vocab_size, emb_dim], dtype=tf.float32)
dec_embeddings = tf.get_variable(
    "dec_embeddings", [dec_vocab_size, emb_dim], dtype=tf.float32)
saver = tf.train.Saver({"enc_embeddings": enc_embeddings, "dec_embeddings": dec_embeddings})
enc_as_op = enc_embeddings.assign(enc_emb_ph)
dec_as_op = dec_embeddings.assign(dec_emb_ph)

sess = tf.Session()
sess.run(enc_as_op, feed_dict={enc_emb_ph: enc_emb_l})
sess.run(dec_as_op, feed_dict={dec_emb_ph: dec_emb_l})

saver.save(sess, finished_files_dir + "embeddings")
