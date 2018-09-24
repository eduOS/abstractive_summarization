# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function
from __future__ import absolute_import
from __future__ import division

from collections import Counter
import pymongo
import pickle
from repos_title import phrase_title


def remake_vocab():

    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    mydb = myclient["mydatabase"]
    mycol = mydb["bytecup"]

    g = mycol.find({"pos_tags": {"$exists": True}})
    pos_word_enc_counter = Counter({})
    pos_word_dec_counter = Counter({})
    stem_counter = Counter({})
    ner_counter = Counter({})
    pos_counter = Counter({})

    for i in g:
        title = i['orig_title']
        # (p, s[0].lower(), s[1])
        info_title = phrase_title(title)
        pos_title = [i[1] for i in info_title]
        title_pos = [i[2] for i in info_title]
        phrase_num = [i[0] for i in info_title]
        mycol.update(
            {"new_id": i['new_id']},
            {
                "$set": {
                    "pos_tag_title": pos_title,
                    "title_pos": title_pos,
                    "title_phrase_indices": phrase_num,
                }
            }, upsert=False)

        pos_word_enc_counter.update(Counter(i["pos_tag_words"]))
        pos_word_dec_counter.update(Counter(title))

        pos_counter.update(Counter(i["pos_tags"]))
        ner_counter.update(Counter(i["ner_tags"]))
        stem_counter.update(Counter(i["stem"]))

    enc_dict = dict(pos_word_enc_counter)
    enc_vocab = sorted(list(zip(enc_dict.keys(), enc_dict.values())), key=lambda x: x[1], reverse=True)
    dec_dict = dict(pos_word_dec_counter)
    dec_vocab = sorted(list(zip(dec_dict.keys(), dec_dict.values())), key=lambda x: x[1], reverse=True)
    pos_dict = dict(pos_counter)
    pos_vocab = sorted(list(zip(pos_dict.keys(), pos_dict.values())), key=lambda x: x[1], reverse=True)
    ner_dict = dict(ner_counter)
    ner_vocab = sorted(list(zip(ner_dict.keys(), ner_dict.values())), key=lambda x: x[1], reverse=True)
    stem_dict = dict(stem_counter)
    stem_vocab = sorted(list(zip(stem_dict.keys(), stem_dict.values())), key=lambda x: x[1], reverse=True)

    try:
        mycol.insert_one({
            "_id": 'vocab_freq_dict',
            'enc_vocab_freq_dict': enc_vocab,
            'dec_vocab_freq_dict': dec_vocab,
            'stem_vocab_freq_dict': stem_vocab,
            "pos_vocab_freq_dict": pos_vocab,
            "ner_vocab_freq_dict": ner_vocab,
        })
    except:
        enc_f = open('enc_vocab_fre_dict.pkl', 'w')
        dec_f = open('dec_vocab_fre_dict.pkl', 'w')
        stem_f = open('stem_vocab_fre_dict.pkl', 'w')
        pos_f = open('pos_vocab_freq_dict.pkl', 'w')
        ner_f = open('ner_vocab_freq_dict.pkl', 'w')
        pickle.dump(enc_vocab, enc_f)
        pickle.dump(dec_vocab, dec_f)
        pickle.dump(stem_vocab, stem_f)
        pickle.dump(pos_vocab, pos_f)
        pickle.dump(ner_vocab, ner_f)
