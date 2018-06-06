# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function
from __future__ import absolute_import
from __future__ import division

from cntk.tokenizer import JiebaTokenizer, text2charlist
from cntk.cleanser import Cleanser
from cntk.standardizer import Standardizer

tokenizer = JiebaTokenizer()
standardizer = Standardizer()
cleanser = Cleanser()

punc_kept = u" ；;。!！,：(（）:?)《》、？，%~"


def sourceline2wordsorchars(line, _char=False, with_digits=True):
    if with_digits:
        line = standardizer.set_sentence(line).fwidth2hwidth().to_lowercase().sentence
    else:
        line = standardizer.set_sentence(line).fwidth2hwidth().to_lowercase().digits().sentence
    if _char:
        words_chars = text2charlist(line)
    else:
        words_chars = tokenizer.sentence2words(line)
    line = " ".join(words_chars)
    line = cleanser.set_sentence(line).delete_useless().sentence
    words_chars = [w for w in line.split() if w]
    return words_chars
