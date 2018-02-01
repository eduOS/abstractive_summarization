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

import numpy as np
import glob


def cal_max_performance(content, reference):
    #  the maximum chars can be copied(in the reference and in the content/len(reference)): 62.2%
    lengths = 0
    assert (len(reference) > len(content))

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

    return lengths_0 / lengths_1


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

    return lengths_0 / lengths_1


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

        scores = []
        for c, r, h in zip(cont, ref, hypo):
            score = calc_score(c, r, h)
            scores.append(score)

        scores = np.transpose(np.array(scores))

        print("the maximum chars can be copied(in the reference and in the content/len(reference)):")
        print(np.mean(scores[0]))
        print(np.std(scores[0]))
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
