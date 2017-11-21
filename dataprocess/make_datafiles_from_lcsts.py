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
from cntk.tokenizer import JiebaTokenizer
from cntk.constants.punctuation import Punctuation
from cntk.standardizer import Standardizer
import re
tokenizer = JiebaTokenizer()
standardizor = Standardizer()
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

END_TOKENS = Punctuation.SENTENCE_DELIMITERS

# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

all_train_urls = "url_lists/all_train.txt"
all_val_urls = "url_lists/all_val.txt"
all_test_urls = "url_lists/all_test.txt"

cnn_tokenized_stories_dir = "cnn_stories_tokenized"
dm_tokenized_stories_dir = "dm_stories_tokenized"
finished_files_dir = "./finished_files/"
chunks_dir = os.path.join(finished_files_dir, "chunked")

# These are the number of .story files we expect there to be in
# cnn_stories_dir and dm_stories_dir
num_expected_cnn_stories = 92579
num_expected_dm_stories = 219506

VOCAB_SIZE = 200000
CHUNK_SIZE = 1000  # num examples per chunk, for the chunked data


def read_text_file(text_file):
    lines = []
    with open(text_file, "r", 'utf-8') as f:
        for line in f:
            lines.append(line.strip())
    return lines


def process_line(line):
    line = standardizor.set_sentence(line).standardize('all').digits().sentence
    lst = tokenizer.sentence2words(line, punc=False)
    return lst


def get_pairs_from_lcsts(filePath, segment=True):
    """
    both should be segmented
    """

    # training set
    # f     = open('./dataset/LCSTS/PART_I/PART_full.txt', 'r')
    f = open(filePath, 'r', 'utf-8')

    line = f.readline().strip()
    lines = 0
    while line:
        if line == '<summary>':
            summary = f.readline().strip()
            if segment:
                summary = process_line(summary)

            f.readline()
            f.readline()
            text = f.readline().strip()
            if segment:
                text = process_line(text)

            pair = (text, summary)
            yield pair
            lines += 1
            if lines % 200000 == 0:
                print(lines)
        line = f.readline().strip()
    print(lines)


def write_to_txt(source_path, out_file, makevocab=False, max_length=100000):
    """Reads the tokenized .story files corresponding to the urls listed in the
    url_file and writes them to a out_file."""

    if makevocab:
        vocab_counter = collections.Counter()

    file_num = 0
    length = 0

    writer = open(out_file + "_" + str(file_num), 'w', 'utf-8')

    for art_tokens, abs_tokens in get_pairs_from_lcsts(source_path):
        # Write to file
        if length >= max_length:
            file_num += 1
            writer.close()
            writer = open(out_file + "_" + str(file_num), 'w', 'utf-8')
            length = 0

        writer.write("ART: " + " ".join(art_tokens)+"\n")
        writer.write("ABS: " + " ".join(abs_tokens)+"\n")
        writer.write("\n")
        length += 1

        if length % (max_length / 10) == 0:
            writer.flush()

        # Write the vocab to file, if applicable
        if makevocab:
            abs_tokens = [
                t for t in abs_tokens if t not in [
                    SENTENCE_START, SENTENCE_END]]
            # remove these tags from vocab
            tokens = art_tokens + abs_tokens
            tokens = [t.strip() for t in tokens]  # strip
            tokens = [t for t in tokens if t != ""]  # remove empty
            vocab_counter.update(tokens)

    # write vocab to file
    if makevocab:
        print("Writing vocab file...")
        with open(
            os.path.join(finished_files_dir, "vocab"), 'w', 'utf-8'
        ) as writer:
            for word, count in vocab_counter.most_common(VOCAB_SIZE):
                writer.write(word + ' ' + str(count) + '\n')
        print("Finished writing vocab file")


def check_num_stories(stories_dir, num_expected):
    num_stories = len(os.listdir(stories_dir))
    if num_stories != num_expected:
        raise Exception(
            "stories directory %s contains %i files but should contain %i" %
            (stories_dir, num_stories, num_expected))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("USAGE: python make_datafiles.py <source_dir>")
        sys.exit()
    source_dir = sys.argv[1]

    # Create some new directories
    if not os.path.exists(finished_files_dir):
        os.makedirs(finished_files_dir)

    # Run stanford tokenizer on both stories dirs, outputting to tokenized
    # stories directories

    # Read the tokenized stories, do a little postprocessing then write to bin
    # files
    write_to_txt(
        source_dir+"PART_III.txt",
        os.path.join(finished_files_dir, "test.txt")
    )
    write_to_txt(
        source_dir+"PART_II.txt",
        os.path.join(finished_files_dir, "val.txt")
    )
    write_to_txt(
        source_dir+"PART_I.txt",
        os.path.join(finished_files_dir, "train.txt"),
    )

    # Chunk the data. This splits each of train.bin, val.bin and test.bin into
    # smaller chunks, each containing e.g. 1000 examples, and saves them in
    # finished_files/chunks
