import numpy
import cPickle as pkl
# import ipdb
import sys
from cntk.tokenizer import text2charslist
from cntk.tokenizer import JiebaTokenizer
from cntk import Standardizor

from collections import OrderedDict
tokenizer = JiebaTokenizer()
standardizor = Standardizor()

"""
build vocabulary from
"""


def main():
    mode = sys.argv[1]
    word_freqs = OrderedDict()
    worddict = OrderedDict()
    if mode == "word":
        worddict['<s>'] = 0
        worddict['</s>'] = 0
        worddict['[PAD]'] = 0
        worddict['[UNK]'] = 1
        worddict['[STOP]'] = 0
        worddict['[START]'] = 1

    assert len(sys.argv) > 3, "numbers of args must be more then 4"
    for filename in sys.argv[2:-1]:
        print('Processing', filename)
        with open(filename, 'r') as f:
            for line in f:
                if line.startswith("<"):
                    continue
                line = standardizor.set_sentence(line).standardize('all').sentence
                if mode == "word":
                    words_in = tokenizer(line, punc=False)
                if mode == "char":
                    words_in = text2charslist(line)
                for w in words_in:
                    if w not in word_freqs:
                        word_freqs[w] = 0
                    word_freqs[w] += 1

    words = word_freqs.keys()
    freqs = word_freqs.values()
    # print freqs
    # ipdb.set_trace()
    sorted_idx = numpy.argsort(freqs)
    # print sorted_idx
    sorted_words = [words[ii] for ii in sorted_idx[::-1]]

    for ii, ww in enumerate(sorted_words):
        worddict[ww] = ii

    with open('%s.pkl' % sys.argv[-1], 'wb') as f:
        pkl.dump(worddict, f)

    print('Done')

if __name__ == '__main__':
    main()
