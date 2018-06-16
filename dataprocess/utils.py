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
    line = cleanser.set_sentence(line).delete_useless().sentence
    if _char:
        words_chars = text2charlist(line)
    else:
        words_chars, sen_labels = tokenizer.text2words(line, dim=1, sen_lab=True)
        assert len(words_chars) == len(sen_labels), line
    if not _char:
        return words_chars, sen_labels
    return words_chars
