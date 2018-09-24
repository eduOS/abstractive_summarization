# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function
from __future__ import absolute_import
from __future__ import division

from nltk.parse.corenlp import CoreNLPParser
from itertools import chain
from utils import pos_repos_tag, tokenize_add_prio
from utils import traverse_tree
from utils import debug_line
import nltk

tokenize = CoreNLPParser(url='http://localhost:9000').tokenize
pos_tagger = CoreNLPParser(url='http://localhost:9000', tagtype='pos')
ner_tagger = CoreNLPParser(url='http://localhost:9000', tagtype='ner')

pattern = r"""
NP: {<DT|PRP\$|CD>?<JJ.?>*(<NN|NNS>|<NE>|<NNP.?>+)}
    {<NN.?>+(<POS>|')<JJ.?>?<NN.?>}
VBD: {<VBD>}
IN: {<IN>}
JJ: {<RB>?<JJ>}
"""
NPChunker = nltk.RegexpParser(pattern)


def phrase_title(ori_title, debug=0):
    sents_pos, tagged_sents = tokenize_add_prio([ori_title], tokenize, pos_tagger, ner_tagger, is_debug=0, log_time=1)
    if debug:
        debug_line('sents_pos', str(sents_pos), "green")
    retagged_sent = pos_repos_tag(tagged_sents, is_debug=0, log_time=1)[0]

    start = -1
    try:
        result = NPChunker.parse(retagged_sent)
    except Exception as e:
        print(str(retagged_sent))
        print()
        print(str(result))
        print()
        print(e)
        return None
    chunked = list(traverse_tree(result, 2, is_debug=0))
    phrase_mark = list(chain.from_iterable(
        [len(c)*[j+start+1]
            for j, c in enumerate(chunked)]))

    assert len(sents_pos[0]) == len(phrase_mark), "sent_pos not equal to phrase mark len"
    info_title = [(p, s[0].lower(), s[1]) for p, s in zip(phrase_mark, sents_pos[0])]

    if debug:
        debug_line("info_title", str(info_title), color="green")
    return info_title
