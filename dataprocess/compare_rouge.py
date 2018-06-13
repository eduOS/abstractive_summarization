# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function
from __future__ import absolute_import
from __future__ import division

from gan_utils import rouge_l
from codecs import open


def load_textfiles(reference, hypothesis):
    hypo = [h.strip() for h in hypothesis]
    # take out newlines before creating dictionary
    refs = [r.strip() for r in reference]
    # sanity check that we have the same number of references as hypothesis
    if len(hypo) != len(refs):
        raise ValueError(
            "There is a sentence number mismatch between the inputs")
    return refs, hypo


if __name__ == '__main__':
    # Feed in the directory where the hypothesis summary and true summary is
    # stored
    segment = True
    hyp_file = ['./decoded.txt']
    ref_file = ['./reference.txt']

    miss = 0.0
    right = 0.0
    wrong = 0.0
    num_files = 0
    r_l = []
    for reference_file, hypothesis_file in zip(ref_file, hyp_file):
        num_files += 1

        with open(reference_file, 'r', 'utf-8') as rf:
            reference = rf.readlines()

        with open(hypothesis_file, 'r', 'utf-8') as hf:
            hypothesis = hf.readlines()

        ref, hypo = load_textfiles(reference, hypothesis)
        for r, h in zip(ref, hypo):
            r_l.append(rouge_l(r, h))

        print('The average rouge_l is %s' % (sum(r_l)/len(r_l)))
