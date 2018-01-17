# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
from codecs import open
from collections import defaultdict as dd
import random
import sys

# generate checkpoint from vocab and embeddings


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
        v, c = vocab_f.readline().strip().split()
        v = v.strip()
        if v in v_l:
            raise "%s is seen previously" % v
        new_embed_l.append(doed[v])
        v_l.append(v)
        count += 1
        if count % 2000 == 0:
            print(count)

    return new_embed_l, emb_dim

vocab_path, embed_path, vocab_size = sys.argv[1], sys.argv[2], int(sys.argv[3])
emb_l, emb_dim = read_from_file(vocab_path, embed_path, vocab_size)
# vocab_size = 100000
# emb_dim = 300
# emb_l = [[1.0] * emb_dim] * vocab_size

emb_ph = tf.placeholder(
    tf.float32, [vocab_size, emb_dim], name='embeddings')

# var_name = 'generator/seq2seq/embeddings'
var_name = 'embeddings'
embeddings = tf.get_variable(
    var_name, [vocab_size, emb_dim], dtype=tf.float32)
saver = tf.train.Saver({"embeddings": embeddings})
as_op = embeddings.assign(emb_ph)

sess = tf.Session()
sess.run(as_op, feed_dict={emb_ph: emb_l})
saver.save(sess, "./temp/embeddings")
