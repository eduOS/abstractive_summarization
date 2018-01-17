import numpy
# import ipdb
import sys
from cntk.tokenizer import text2charlist
from cntk.tokenizer import JiebaTokenizer
from cntk.cleanser import Cleanser
from utils import sourceline2words
from cntk.standardizer import Standardizer
from codecs import open

from collections import OrderedDict
tokenizer = JiebaTokenizer()
standardizor = Standardizer()
cleanser = Cleanser()

"""
build vocabulary from
"""

# python build_dictionary ['word', 'char'] (./filepathes) output_dir


def main():
    mode = sys.argv[1]
    word_freqs = OrderedDict()
    # worddict['[PAD]'] = sys.maxint
    # worddict['[UNK]'] = sys.maxint
    # worddict['[STOP]'] = sys.maxint
    # worddict['[START]'] = sys.maxint

    assert len(sys.argv) > 3, "numbers of args must be more then 4"
    for filename in sys.argv[2:-1]:
        print('Processing', filename)
        with open(filename, 'r', 'utf-8') as f:
            for line in f:
                if not line or line.strip().startswith("<"):
                    continue
                if mode == "word":
                    words_in = sourceline2words(line)
                elif mode == "char":
                    words_in = text2charlist(line)
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
    must_include = ['[PAD]', '[UNK]', '[STOP]', '[START]'] if mode == 'word' else ['[PAD]', '[UNK]']
    sorted_words = must_include + [words[ii] for ii in sorted_idx[::-1]]
    sorted_freq = len(must_include) * [1] + [freqs[ii] for ii in sorted_idx[::-1]]

    with open('%s' % sys.argv[-1], 'w', 'utf-8') as f:
        for ii, ww in zip(sorted_freq, sorted_words):
            f.write(ww + " " + str(ii) + "\n")

    print('Done')

if __name__ == '__main__':
    main()
