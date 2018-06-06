# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function
from __future__ import absolute_import
from __future__ import division

from cntk.tokenizer import JiebaTokenizer
from cntk.cleanser import Cleanser
from cntk.standardizer import Standardizer

tokenizer = JiebaTokenizer()
standardizer = Standardizer()
cleanser = Cleanser()

punc_kept = u" ；;。!！,：(（）:?)《》、？，%~"


def sourceline2words(line, with_digits=True):
    if with_digits:
        line = standardizer.set_sentence(line).fwidth2hwidth().to_lowercase().sentence
    else:
        line = standardizer.set_sentence(line).fwidth2hwidth().digits().to_lowercase().sentence
    words = tokenizer.sentence2words(line)
    line = " ".join(words)
    line = cleanser.set_sentence(line).delete_useless().sentence
    words = [w for w in line.split() if w]
    return words
