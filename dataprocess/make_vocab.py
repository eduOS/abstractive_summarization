# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function
from __future__ import absolute_import
from __future__ import division

import pymongo


enc_vocab_size = 50000
dec_vocab_size = 50000


myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["mydatabase"]
mycol = mydb["bytecup2018"]

whole_vocab = list(mycol.find({"_id": 'vocab_freq_dict'}))[0]


def main():
   enc_vocab = whole_vocab['enc_vocab_freq_dict']
   dec_vocab = whole_vocab['dec_vocab_freq_dict']
   lemma_vocab = whole_vocab['lemma_vocab_freq_dict']

   dec_vocab =
