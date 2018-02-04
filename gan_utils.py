# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function
from __future__ import absolute_import
from __future__ import division

import math
import datetime
from termcolor import colored
from os.path import join as join_path


def check_rouge(sess, decoder, best_rouge, val_batcher, val_path, val_saver, global_step, sample_rate=0):
    """
    model: the decoder
    val_batcher: the gen_val_batcher
    """
    saved = False
    ave_rouge = decoder.bs_decode(val_batcher, save2file=False, single_pass=True, sample_rate=sample_rate)
    if ave_rouge > best_rouge:
        val_saver.save(sess, val_path, global_step=global_step)
        best_rouge = ave_rouge
        saved = True
        print('Found new best model with %.3f evaluation rouge-l. Saving to %s %s' %
              (best_rouge, val_path,
               datetime.datetime.now().strftime("on %m-%d at %H:%M")))
    return ave_rouge, best_rouge, saved


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


def rouge_l(samples, references, beta=1.2, rs=None):
    """
    samples: list of list,
    references: list of list
    """
    prec = []
    rec = []
    scores = []
    for n, (s, r) in enumerate(zip(samples, references)):
        if len(s) == 0 or len(r) == 0:
            prec.append(0)
            rec.append(0)
            continue
        lcs = my_lcs(s, r)
        prec.append(lcs/float(len(s)))
        rec.append(lcs/float(len(r)))

    for p, r in zip(prec, rec):
        if(p != 0 and r != 0):
            score = ((1 + beta**2) * p * r) / float(r + beta**2 * p)
        else:
            score = 0.0
        scores.append(score)
    return scores


def save_ckpt(sess, model, decoder, best_loss, best_rouge, model_dir, model_saver,
              loss_batcher, rouge_batcher, rouge_dir, rouge_saver, global_step, sample_rate):
    """
    save model to model dir or evaluation directory
    the loss batcher is the val beatcher, while the rouge batcher is the test batcher

    """

    saved = False
    val_save_path = join_path(rouge_dir, "best_model")
    model_save_path = join_path(model_dir, "model")

    losses = []
    while True:
        val_batch = loss_batcher.next_batch()
        if not val_batch:
            break
        results_val = model.run_one_batch(sess, val_batch, update=False, gan_eval=True)
        loss_eval = results_val["loss"]
        # why there exists nan?
        if not math.isnan(loss_eval):
            losses.append(loss_eval)
        else:
            print(colored("Encountered a NAN in evaluating GAN loss.", 'red'))
    eval_loss = sum(losses) / len(losses)
    if best_loss is None or eval_loss < best_loss:
        best_loss = eval_loss

    ave_rouge, best_rouge, saved = check_rouge(
        sess, decoder, best_rouge, rouge_batcher,
        val_save_path, rouge_saver, global_step, sample_rate=sample_rate)

    if not saved:
        model_saver.save(sess, model_save_path, global_step=global_step)
        print("Model is saved to" + colored(" %s", 'yellow') % model_dir)

    return eval_loss, best_loss, ave_rouge, best_rouge