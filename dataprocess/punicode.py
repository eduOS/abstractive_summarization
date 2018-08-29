# -*  coding: utf-8 -*-
# tested only for python3.5

from __future__ import unicode_literals, print_function
from __future__ import absolute_import
from __future__ import division

from nltk.parse.corenlp import CoreNLPParser
from unidecode import unidecode
# from codecs import open
import collections
import json
from termcolor import colored
from codecs import open
import os
from os.path import join
from nltk import sent_tokenize
import re
import unicodedata
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.stem import LancasterStemmer
port = PorterStemmer()
wnl = WordNetLemmatizer()
ls = LancasterStemmer()

ENC_VOCAB_SIZE = 500000
DEC_VOCAB_SIZE = 100000

END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', ")"]

finished_files_dir = "./finished_files/"
if not os.path.exists(finished_files_dir):
    os.makedirs(finished_files_dir)

tokenize = CoreNLPParser(url='http://localhost:9000').tokenize
pos_tagger = CoreNLPParser(url='http://localhost:9000', tagtype='pos')
ner_tagger = CoreNLPParser(url='http://localhost:9000', tagtype='ner')

enc_must_include = ['[pad]', '[unk]']
dec_must_include = ['[pad]', '[unk]', '[stop]', '[start]']

# fi = open('./data/dptest.txt', encoding='unicode_escape')


def my_repl(match):
    # how can I add a parameter here?
    uni = match.group('uni')
    if uni:
        if unicodedata.category(uni).startswith("P"):
            # if color:
            #     return colored("[" + unidecode(uni) + "]", color or 'green')
            uni = unidecode(uni)
            return "''" if uni == '"' else uni
        else:
            # if color:
            #     return colored("[" + ' UNK ' + "]", color or 'green')
            return ' u_n_k '


def debug_line(tag, line):
    tag = colored(tag + ': ', 'green')
    default_len = 1000
    if len(line) < 1000:
        ipt = "f"
        if tag:
            print(colored(tag, 'yellow'), end='\t')
    else:
        ipt = input(colored('input the char number to see ', 'yellow') + tag + colored(', f for all chars, default is %s: ' % default_len, 'yellow'))
    # sys.stdout.write('\r')
    if ipt == "f":
        print(line)
    elif not ipt:
        print(line[:default_len], end="")
        print("...")
    elif ipt:
        print(line[:int(ipt)], end="")
        print("...")


def debug_list(list_text, file_name):
    with open(file_name, "w", "utf-8") as fo:
        fo.write("\n".join(list_text))


def fix_missing_period(line):
  """Adds a period to a line that is missing a period"""
  if line == "":
      return line
  if line[-1] in END_TOKENS:
      return line
  return line + " ."


def cut_sentence(line, is_debug=False):
    """
    the existing tools may be better
    """
    line = line.replace("''", '"')
    sentences = re.sub(r'''((?<![A-Z])\.(?=[A-Z][^.])|\.(?=([ '"]+|by))\)*|[;!?:]['"]*)|(?<![\d ])\.(?=\d\.?)''', r'\1\n', line).split('\n')
    sentences = [sentence.strip() for sentence in sentences if len(sentence.strip()) > 1]

    sentences_ = sent_tokenize(line)
    # failed when encountering ; l.U \n .by

    if is_debug:
        for sentence in sentences:
            debug_line("my cut sentence", sentence)
        input('\n\n')
        for sentence in sentences_:  # noqa
            debug_line('corenlp cut', sentence)
        input('\n\n')
    else:
        return sentences


def make_vocab(enc_vocab_counter, dec_vocab_counter):
    print("Writing vocab file...")
    enc_vocab_counter.pop("", "null exists")
    dec_vocab_counter.pop("", "null exists")
    dec_total_vocab = sum(dec_vocab_counter.values())
    enc_total_vocab = sum(enc_vocab_counter.values())
    dec_vocab_size = dec_total_vocab if DEC_VOCAB_SIZE > dec_total_vocab else DEC_VOCAB_SIZE
    enc_vocab_size = dec_total_vocab if ENC_VOCAB_SIZE > enc_total_vocab else ENC_VOCAB_SIZE
    input("Total decoder vocab size: %s, encoder vocab size: %s" % (dec_total_vocab, enc_total_vocab))
    # stats and input vocab size

    dec_most_common = dec_vocab_counter.most_common(dec_vocab_size-len(dec_must_include))
    for dec_key in dec_most_common:
        enc_vocab_counter.pop(dec_key, None)

    enc_most_common = enc_vocab_counter.most_common(enc_vocab_size-len(enc_must_include)-dec_vocab_size)

    acc_p = 0
    dec_writer = open(join(finished_files_dir, "dec_vocab"), 'w', 'utf-8')
    enc_writer = open(join(finished_files_dir, "enc_vocab"), 'w', 'utf-8')

    for word, count in dec_most_common:
        acc_p += (count / dec_total_vocab)
        # if acc_p > 0.96:
        #     break
        dec_writer.write(word + ' ' + str(count) + " " + str(acc_p) + '\n')
        enc_writer.write(word + ' ' + str(count) + " " + str(acc_p) + '\n')
    for mi in dec_must_include:
        dec_writer.write(mi + ' ' + "1" + " 0.0" + '\n')
    dec_writer.close()
    print("Finished writing dec_vocab file")
    for mi in enc_must_include:
        enc_writer.write(mi + ' ' + "1" + " 0.0" + '\n')

    acc_p = 0
    for word, count in enc_most_common:
        acc_p += (count / enc_total_vocab)
        enc_writer.write(word + ' ' + str(count) + " " + str(acc_p) + '\n')
    enc_writer.close()
    print("Finished writing enc_vocab file")


def bytes2unicode(line, is_debug=False):
    line = line.replace(b'\\"', b"''")
    if is_debug:
        debug_line('replaced "', line)

    line = line.decode('unicode_escape', errors='ignore')
    if is_debug:
        debug_line('escape decoded unicode', line)

    line = re.sub(pattern=r'(?P<uni>[^\x00-\x7f])', repl=my_repl, string=line)
    if is_debug:
        debug_line('unidecoded', line)
    return line


def map_tfidf(normalized_sentences, is_debug=True):
    # lemmanization and score
    pass


def word_normalize(tagged_sentences, is_debug=False):
    normalized_sentences = []
    for tagged_sentence in tagged_sentences:
        # lower case the first word
        tagged_sentence[0][0].lower()
        debug_line('lowered first word in function', str(tagged_sentence))
        # ported = [port.stem(i) for i in list(map(lambda x: x[0], tagged_sentence))]
        # lsed = [ls.stem(i) for i in list(map(lambda x: x[0], tagged_sentence))]
        wnled = [wnl.lemmatize(w, pos='a' if p[0].lower() == 'j' else p[0].lower())
                 if p[0].lower() in ['j', 'r', 'n', 'v'] else w
                 for w, p in list(map(lambda x: (x[0], x[1]), tagged_sentence))]
        normalized_sentences.append(wnled)

        if is_debug:
            debug_line("tagged sentence", str(tagged_sentence))
            # debug_line("sent_stem ported", str(ported))
            # debug_line("sent_stem lancester", str(lsed))
            debug_line("sent_lemma wnled", str(wnled))
            input('\n')
    return normalized_sentences


def tokenize_add_prio(sentences, is_debug=False):

    tagged_sentences = []
    tokenized_sentences = [list(tokenize(sentence)) for sentence in sentences]
    sentences_pos = [pos_tagger.tag(sentence) for sentence in tokenized_sentences]
    sentences_ner = [ner_tagger.tag(sentence) for sentence in tokenized_sentences]
    for sp, sn in zip(sentences_pos, sentences_ner):
        mapped = list(map(
            lambda i_y: (
                i_y[1][0][0].lower() if i_y[0] < 2 and i_y[1][0][1] not in ['NNP', 'NNPS'] else i_y[1][0][0],
                i_y[1][0][1],
                i_y[1][1][1]),
            enumerate(zip(sp, sn))))
        tagged_sentences.append(mapped)
        if is_debug:
            debug_line("sentence pos", str(sp))
            debug_line("sentence ner", str(sn))
            debug_line("sentence combined", str(mapped))
            input('\n')
    return tagged_sentences


def process_title(title, is_debug=False):
    if 'u_n_k' in title:
        return None
    else:
        title = title.replace("''", '"')
        title = tokenize(title)
        title = ' '.join(list(title)).lower()
    return title


def delete_unk_sentences(sentences):
    """
    delete by hand, if too many unks
    """
    return


def read_origin(fi, is_debug=False):
    line = fi.readline()
    if not line:
        return None
    if is_debug:
        debug_line('origin', line)
    return line


def sent_filter(sentences, debug=False):
    sentences = list(filter(lambda x: len(x) > 2, sentences))


def load_json(line, is_debug=False):
    _json = json.loads(line, strict=False)
    return _json['id'], _json['title'], _json['content']


def main(makevocab=True):
    if makevocab:
        enc_vocab_counter = collections.Counter()
        dec_vocab_counter = collections.Counter()
    fi = open('./data/dptest.txt', 'rb')

    while(True):
        line = read_origin(fi, is_debug=True)
        if not line:
            break

        line = bytes2unicode(line)
        _id, title, content = load_json(line)

        title = process_title(title)
        if not title:
            continue
        sentences = cut_sentence(content, is_debug=False)
        # sentences = delete_unk_sentences(sentences, is_debug=True)

        # sentences = sent_filter(sentences, debug=False)
        tagged_sentences = tokenize_add_prio(sentences, is_debug=False)
        normalized_sentences = word_normalize(tagged_sentences, is_debug=True)
        # debug_line('the origin sentence changed?', str(tagged_sentences))
        # the the first word of every sentence in tagged_sentences should be
        # lower cased
        # scored_sentences = map_tfidf(normalized_sentences, is_debug=True)

        input('\n\n')
        if makevocab:
            enc_vocab_counter.update(content.split(" "))
            dec_vocab_counter.update(title.split(" "))

    # write vocab to file
    if makevocab:
        make_vocab(enc_vocab_counter, dec_vocab_counter)


def test():
    main(makevocab=False)

if __name__ == '__main__':
    test()
