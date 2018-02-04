#!/usr/bin/env python
#
#     the maximum chars can be copied(in the reference and in the content/len(reference)): 62.2%
#
#     performance(in the content and in the hypothesis/hypothesis):
#
#     rate of correction(in the content, in the reference and in the hypothesis/in the content, in the reference):
#
#     rate of mistakenly copied(in the content, not in the reference, but in the hypothesis/hypothesis):
#     rate of missing(in the content, in the reference but not in the hypothesis/in the content, in the reference):

from __future__ import division
import numpy as np
import glob


def cal_max_performance(content, reference):
    #  the maximum chars can be copied(in the reference and in the content/len(reference)): 62.2%
    lengths = 0
    assert (len(reference) < len(content))

    for ch in reference:
        if ch in content:
            lengths += 1

    return lengths / len(reference)


def cal_performance(content, hypothesis):
    # performance(in the content and in the hypothesis/hypothesis):
    lengths = 0
    assert (len(content) > len(hypothesis))

    for ch in hypothesis:
        if ch in content:
            lengths += 1

    return lengths / len(hypothesis)


def cal_correction(content, reference, hypothesis):
    # rate of correction(in the content, in the reference and in the
    # hypothesis/in the content, in the reference):
    lengths_0 = 0
    lengths_1 = 0
    assert (len(content) > len(hypothesis))

    for ch in reference:
        if ch in content and ch in hypothesis:
            lengths_0 += 1
        if ch in content:
            lengths_1 += 1

    return lengths_0 / lengths_1 if lengths_1 else 0


def cal_mistake(content, reference, hypothesis):
    # rate of mistakenly copied(in the content, not in the reference, but in
    # the hypothesis/hypothesis):
    lengths_0 = 0
    assert (len(content) > len(hypothesis))

    for ch in hypothesis:
        if ch in content and ch not in reference:
            lengths_0 += 1

    return lengths_0 / len(hypothesis)


def cal_missing(content, reference, hypothesis):
    # rate of missing(in the content, in the reference but not in the
    # hypothesis/in the content, in the reference):
    lengths_0 = 0
    lengths_1 = 0
    assert (len(content) > len(hypothesis))

    for ch in reference:
        if ch in content and ch not in hypothesis:
            lengths_0 += 1
        if ch in content:
            lengths_1 += 1

    return lengths_0 / lengths_1 if lengths_1 else 0


def calc_score(content, reference, hypothesis):
    """
    Compute ROUGE-L score given one candidate and references for an image
    :param candidate: str : candidate sentence to be evaluated
    :param refs: list of str : COCO reference sentences for the particular image to be evaluated
    :returns score: int (ROUGE-L score for the candidate evaluated against references)
    """
    # split into tokens
    token_c = content.split(" ")
    token_c = filter(lambda x: x != " ", token_c)

    token_r = reference.split(" ")
    token_r = filter(lambda x: x != " ", token_r)

    token_h = hypothesis.split(" ")
    token_h = filter(lambda x: x != " ", token_h)

    missing = cal_missing(token_c, token_r, token_h)
    mistake = cal_mistake(token_c, token_r, token_h)
    correct = cal_correction(token_c, token_r, token_h)
    perform = cal_performance(token_c, token_h)
    max_per = cal_max_performance(token_c, token_r)

    return np.array([max_per, perform, correct, missing, mistake])


def my_lcs(string, sub):
    """
    Calculates longest common subsequence for a pair of tokenized strings
    :param string : list of str : tokens from a string split using whitespace
    :param sub : list of str : shorter string, also split using whitespace
    :returns: length (list of int): length of the longest common subsequence between the two strings

    Note: my_lcs only gives length of the longest common subsequence, not the actual LCS
    """
    if(len(string) < len(sub)):
        sub, string = string, sub

    lengths = [[0 for i in range(0, len(sub)+1)] for j in range(0, len(string)+1)]

    for j in range(1, len(sub)+1):
        for i in range(1, len(string)+1):
            if(string[i-1] == sub[j-1]):
                lengths[i][j] = lengths[i-1][j-1] + 1
            else:
                lengths[i][j] = max(lengths[i-1][j], lengths[i][j-1])

    return lengths[len(string)][len(sub)]


def calc_mean_rouge_l_pres(contents, references):
    """
    samples: list of list,
    references: list of list
    """
    prec = []
    rec = []
    for n, (s, r) in enumerate(zip(references, contents)):
        token_c = r.split(" ")
        token_c = filter(lambda x: x != " ", token_c)

        token_r = s.split(" ")
        token_r = filter(lambda x: x != " ", token_r)

        if len(token_r) == 0 or len(token_c) == 0:
            prec.append(0)
            rec.append(0)
            continue
        lcs = my_lcs(token_r, token_c)
        prec.append(lcs/float(len(token_r)))
    return np.array(prec)


def load_textfiles(content, reference, hypothesis):
    cont = [lines.strip() for (idx, lines) in enumerate(content)]
    hypo = [lines.strip() for (idx, lines) in enumerate(hypothesis)]
    refe = [lines.strip() for (idx, lines) in enumerate(reference)]
    # take out newlines before creating dictionary
    # sanity check that we have the same number of references as hypothesis
    if len(hypo) != len(refe) != len(cont):
        raise ValueError(
            "There is a sentence number mismatch between the inputs")
    return cont, refe, hypo


if __name__ == '__main__':
    # Feed in the directory where the hypothesis summary and true summary is
    # stored
    hyp_file = glob.glob('hypothesis/*')
    ref_file = glob.glob('reference/*')
    con_file = glob.glob('condition/*')

    ROUGE_L = 0.
    num_files = 0
    for content_file, reference_file, hypothesis_file in zip(con_file, ref_file, hyp_file):
        num_files += 1

        with open(content_file) as rf:
            content = rf.readlines()

        with open(reference_file) as rf:
            reference = rf.readlines()

        with open(hypothesis_file) as hf:
            hypothesis = hf.readlines()

        cont, ref, hypo = load_textfiles(content, reference, hypothesis)
        rlprec = calc_mean_rouge_l_pres(cont, ref)

        scores = []
        for c, r, h in zip(cont, ref, hypo):
            score = calc_score(c, r, h)
            scores.append(score)

        scores = np.transpose(np.array(scores))

        print("the maximum chars can be copied(in the reference and in the content/len(reference)):")
        print(np.mean(scores[0]))
        print(np.std(scores[0]))
        print("the rouge-l precision(lcs(in the reference and in the content)/len(reference)):")
        print(np.mean(rlprec))
        print(np.std(rlprec))
        print("performance(in the content and in the hypothesis/hypothesis):")
        print(np.mean(scores[1]))
        print(np.std(scores[1]))
        print("rate of correction(in the content, in the reference and in the hypothesis/in the content, in the reference):")
        print(np.mean(scores[2]))
        print(np.std(scores[2]))
        print("rate of missing(in the content, in the reference but not in the hypothesis/in the content, in the reference):")
        print(np.mean(scores[3]))
        print(np.std(scores[3]))
        print("rate of mistakenly copied(in the content, not in the reference, but in the hypothesis/hypothesis):")
        print(np.mean(scores[4]))
        print(np.std(scores[4]))

        # ([max_per, perform, correct, missing, mistake])
