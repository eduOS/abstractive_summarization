# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function
from __future__ import absolute_import
from __future__ import division
import sys
import os
# import hashlib
# import subprocess
import collections
# import tensorflow as tf
# from tensorflow.core.example import example_pb2
import os.path
from codecs import open
from cntk.tokenizer import JiebaTokenizer, text2charlist
from utils import sourceline2words
from cntk.constants.punctuation import Punctuation
from cntk.standardizer import Standardizer
import numpy as np
import time
tokenizer = JiebaTokenizer()
standardizor = Standardizer()
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

END_TOKENS = Punctuation.SENTENCE_DELIMITERS

# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

finished_files_dir = "./finished_files/"

ENC_VOCAB_SIZE = 20000
DEC_VOCAB_SIZE = 500000

start = time.time()
enc_must_include = ['[PAD]', '[UNK]']
dec_must_include = ['[PAD]', '[UNK]', '[STOP]', '[START]']


def read_text_file(text_file):
    lines = []
    with open(text_file, "r", 'utf-8') as f:
        for line in f:
            lines.append(line.strip())
    return lines


def process_line(line):
    return sourceline2words(line)


len_art = []
len_abs = []

log_file = open('corpus_log', 'a', 'utf-8')


def get_pairs_from_lcsts(filePath, enc_segment=False, dec_segment=False):
    """
    both should be segmented if segment is true
    """

    # training set
    # f     = open('./dataset/LCSTS/PART_I/PART_full.txt', 'r')
    f = open(filePath, 'r', 'utf-8')

    line = f.readline().strip()
    lines = 0
    flag = 0
    while line:
        if line == '<summary>':
            summary = f.readline().strip()
            if flag != 0:
                print(summary)
                raise Exception("something went wrong in %s" % filePath)
            if f.readline().strip() != "</summary>":
                print(summary)
                raise Exception("something went wrong in %s" % filePath)
            flag = 1

            if dec_segment:
                summary = process_line(summary)
            else:
                summary = text2charlist(summary)

            line = f.readline().strip()

        elif line == '<short_text>':
            text = f.readline().strip()
            if flag != 1:
                print(summary)
                raise Exception("something went wrong in %s" % filePath)
            if f.readline().strip() != "</short_text>":
                print(text)
                raise Exception("something went wrong in %s" % filePath)
            flag = 0

            if enc_segment:
                text = process_line(text)
            else:
                text = text2charlist(text)
            line = f.readline().strip()

        else:
            line = f.readline().strip()
            continue

        if flag == 0:
            dont_yield = 0
            abs_l = len(summary)
            len_abs.append(abs_l)
            if abs_l <= 3:
                log_file.write(filePath)
                log_file.write('\n')
                log_file.write('summary')
                log_file.write('\n')
                log_file.write(" ".join(summary))
                log_file.write('\n')
                log_file.write('\n')
                dont_yield = 1
            art_l = len(text)
            len_art.append(art_l)
            if art_l < 20:
                log_file.write(filePath)
                log_file.write('\n')
                log_file.write('text')
                log_file.write('\n')
                log_file.write(" ".join(text))
                log_file.write('\n')
                log_file.write('\n')
                dont_yield = 1
            pair = (text, summary)
            if dont_yield:
                continue
            else:
                lines += 1
                if lines % 200000 == 0:
                    print(lines)
                yield pair

    f.close()
    print(lines)


def write_to_txt(source_path, out_file, makevocab=False, max_length=100000, enc_segment=False, dec_segment=False):
    """Reads the tokenized .story files corresponding to the urls listed in the
    url_file and writes them to a out_file."""

    if makevocab:
        enc_vocab_counter = collections.Counter()
        dec_vocab_counter = collections.Counter()

    file_num = 0
    length = 0

    writer = open(out_file + "_" + str(file_num), 'w', 'utf-8')

    for art_tokens, abs_tokens in \
            get_pairs_from_lcsts(source_path, enc_segment=enc_segment, dec_segment=dec_segment):
        # Write to file
        if length >= max_length:
            file_num += 1
            writer.close()
            writer = open(out_file + "_" + str(file_num), 'w', 'utf-8')
            length = 0

        writer.write(" ".join(art_tokens) + "\t" + " ".join(abs_tokens) + "\n")
        length += 1

        if length % (max_length / 10) == 0:
            writer.flush()

        # Write the vocab to file, if applicable
        if makevocab:
            abs_tokens = [
                t for t in abs_tokens if t not in [
                    SENTENCE_START, SENTENCE_END]]
            # remove these tags from vocab
            enc_tokens = [t.strip() for t in art_tokens]  # strip
            enc_tokens = [t for t in enc_tokens if t != ""]  # remove empty
            dec_tokens = [t.strip() for t in abs_tokens]  # strip
            dec_tokens = [t for t in dec_tokens if t != ""]  # remove empty
            enc_vocab_counter.update(enc_tokens)
            dec_vocab_counter.update(dec_tokens)

    writer.close()
    # write vocab to file
    if makevocab:
        print("Writing vocab file...")
        total_vocab = sum(enc_vocab_counter.values())
        acc_p = 0
        with open(
            os.path.join(finished_files_dir, "enc_vocab"), 'w', 'utf-8'
        ) as writer:
            for mi in enc_must_include:
                writer.write(mi + ' ' + "1" + " 0.0" + '\n')
            for word, count in enc_vocab_counter.most_common(ENC_VOCAB_SIZE):
                acc_p += (count / total_vocab)
                writer.write(word + ' ' + str(count) + " " + str(acc_p) + '\n')
        print("Finished writing enc_vocab file")

        total_vocab = sum(dec_vocab_counter.values())
        acc_p = 0
        with open(
            os.path.join(finished_files_dir, "dec_vocab"), 'w', 'utf-8'
        ) as writer:
            for mi in dec_must_include:
                writer.write(mi + ' ' + "1" + " 0.0" + '\n')
            for word, count in dec_vocab_counter.most_common(DEC_VOCAB_SIZE):
                acc_p += (count / total_vocab)
                writer.write(word + ' ' + str(count) + " " + str(acc_p) + '\n')
        print("Finished writing dec_vocab file")

    # cumulative probability


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("USAGE: python make_datafiles.py <source_dir>")
        sys.exit()
    source_dir = sys.argv[1]
    enc_segment = False
    dec_segment = True

    # Create some new directories
    if not os.path.exists(finished_files_dir):
        os.makedirs(finished_files_dir)

    # Run stanford tokenizer on both stories dirs, outputting to tokenized
    # stories directories

    # Read the tokenized stories, do a little postprocessing then write to bin
    # files
    write_to_txt(
        source_dir+"PART_III.txt",
        os.path.join(finished_files_dir, "test.txt"),
        enc_segment=enc_segment, dec_segment=dec_segment
    )
    write_to_txt(
        source_dir+"PART_II.txt",
        os.path.join(finished_files_dir, "val.txt"),
        enc_segment=enc_segment, dec_segment=dec_segment
    )
    write_to_txt(
        source_dir+"PART_I.txt",
        os.path.join(finished_files_dir, "train.txt"), makevocab=True,
        enc_segment=enc_segment, dec_segment=dec_segment
    )

    log_file.write("the mean of art: %s" % float(np.mean(len_art)))
    log_file.write('\n')
    log_file.write("the std of art: %s" % float(np.std(len_art)))
    log_file.write('\n')
    log_file.write("the max of art: %s" % float(np.max(len_art)))
    log_file.write('\n')
    log_file.write("the min of art: %s" % float(np.min(len_art)))
    log_file.write('\n')
    log_file.write('\n')

    log_file.write("the mean of abs: %s" % float(np.mean(len_abs)))
    log_file.write('\n')
    log_file.write("the std of abs: %s" % float(np.std(len_abs)))
    log_file.write('\n')
    log_file.write("the max of abs: %s" % float(np.max(len_abs)))
    log_file.write('\n')
    log_file.write("the min of abs: %s" % float(np.min(len_abs)))
    log_file.write('\n')
    log_file.write('\n')
    ts = (time.time() - start) / 3600
    log_file.write('time spent %.4f h' % ts)
    log_file.close()

    # Chunk the data. This splits each of train.bin, val.bin and test.bin into
    # smaller chunks, each containing e.g. 1000 examples, and saves them in
    # finished_files/chunks
