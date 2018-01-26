# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function
from __future__ import absolute_import
from __future__ import division

from cntk.tokenizer import JiebaTokenizer
from cntk.cleanser import Cleanser
from cntk.standardizer import Standardizer

tokenizer = JiebaTokenizer()
standardizor = Standardizer()
cleanser = Cleanser()

punc_kept = u" ；;。!！,：(（）:?)《》、？，%~"


def sourceline2words(line, with_digits=True):
    words = tokenizer.sentence2words(line)
    line = " ".join(words)
    # line = standardizor.set_sentence(line).to_lowercase(verbose=False).fwidth2hwidth(verbose=True).digits(verbose=True).sentence
    if with_digits:
        line = standardizor.set_sentence(line).to_lowercase().sentence
    else:
        line = standardizor.set_sentence(line).to_lowercase().digits().sentence
    line = cleanser.set_sentence(line).delete_useless().sentence
    words = [w for w in line.split() if w]
    return words
