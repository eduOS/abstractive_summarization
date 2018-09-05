# -*  coding: utf-8 -*-
# tested only for python3.5

from __future__ import unicode_literals, print_function
from __future__ import absolute_import
from __future__ import division

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
import time
from itertools import chain

import nltk
from utils import timeit
from nltk.tree import Tree
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.stem import LancasterStemmer
from nltk.corpus import stopwords
from nltk.parse.corenlp import CoreNLPParser

# for spacy
from spacy.en import English, LOCAL_DATA_DIR
import spacy.en
import os

import string
from sklearn.feature_extraction.text import TfidfVectorizer
import multiprocessing
port = PorterStemmer()
wnl = WordNetLemmatizer()
ls = LancasterStemmer()
ENC_VOCAB_SIZE = 500000
DEC_VOCAB_SIZE = 100000
NUM_WORKERS = multiprocessing.cpu_count() * 2

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

pattern = r"""
NP: {<DT|PRP\$|CD>?<JJ.?>*(<NN|NNS>|<NE>|<NNP.?>+)}
    {<NN.?>+(<POS>|')<JJ.?>?<NN.?>}
VBD: {<VBD>}
IN: {<IN>}
JJ: {<RB>?<JJ>}
"""


def dummy_fun(doc):
    return doc

stopwords = stopwords.words('english') + list(string.punctuation)

tfidf = TfidfVectorizer(stop_words=stopwords, analyzer='word', tokenizer=dummy_fun, preprocessor=dummy_fun, token_pattern=None)


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


def debug_line(tag, line, color="green"):
    tag = colored(tag + ':', color)
    default_len = 1500
    if len(line) < 1000:
        ipt = "f"
        if tag:
            print(colored(tag, 'yellow'))
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


@timeit
def cut_sent(line, is_debug=False, log_time=0):
    """
    TODO: combining the existing tools may be better
    """
    if is_debug:
        debug_line("have quote", line)
    line = line.replace("''", '"')
    if is_debug:
        debug_line("replace quote", line)

    sents = re.sub(r'''((?<![A-Z])\.(?=[A-Z][^.])|\.(?=([ '"]+|by))\)*|[;!?:]['"]*)|(?<![\d ])\.(?=\d\.?)''', r'\1\n', line).split('\n')
    sents = [sent.strip() for sent in sents if len(sent.strip()) > 1]

    if is_debug:
        sents_ = sent_tokenize(line)
        # failed when encountering ; l.U \n .by

    if is_debug:
        for sent in sents:
            debug_line("my cut sent", sent)
        input('\n\n')
        for sent in sents_:  # noqa
            debug_line('corenlp cut', sent)
        input('\n\n')
    else:
        return sents


@timeit
def pos_repos_tag(tagged_sents, is_debug=0, log_time=0):
    """
    pos retag according to the ner tag, the named entity are tagged as <NE>
    """

    pos_retagged_sents = []
    for tagged_sent in tagged_sents:
        length = 0
        current_sent = []
        continuous_chunk = []

        for (pos_token, _, p_tag, n_tag) in tagged_sent:
            if n_tag != "O":
                continuous_chunk.append(pos_token)
            else:
                if continuous_chunk:  # if the current chunk is not empty
                    if is_debug:
                        retagged = ("_".join(continuous_chunk), "_NE_")
                        length += len(continuous_chunk)
                    else:
                        retagged = (" ".join(continuous_chunk), "NE")
                        length += len(continuous_chunk)
                    current_sent.append(retagged)
                    continuous_chunk = []
                current_sent.append((pos_token, p_tag))
                length += 1

        # add the last continuous tokens
        if continuous_chunk:
            if is_debug:
                retagged = ("_".join(continuous_chunk), "_NE_")
                length += len(continuous_chunk)
            else:
                retagged = (" ".join(continuous_chunk), "NE")
                length += len(continuous_chunk)
            current_sent.append(retagged)

        if is_debug:
            debug_line("is in debug mode, no steps can be followed, error may occur by _", '', "red")
            debug_line("origial sent", str(tagged_sent))
            debug_line("retagged sent", str(current_sent), 'red')
        assert len(tagged_sent) == length, "length of tagged_sent and current_sent should be the same, but %s and %s\n%s\n%s" % (len(tagged_sent), length, str(tagged_sent), str(current_sent))
        pos_retagged_sents.append(current_sent)

    return pos_retagged_sents


def traverse_tree(tree, depth=float('inf'), is_debug=0):
    """
    Traversing the Tree depth-first,
    yield leaves up to `depth` level.
    """
    for subtree in tree:
        if type(subtree) == Tree:
            if subtree.height() <= depth:
                leaves = ' '.join(subtree.leaves()).split(' ')
                yield list(map(lambda x: x.split('/')[0] if not is_debug else x, leaves))
                traverse_tree(subtree)
        else:
            # the named entity should be separated
            leaves = [leaf.split('/')[0] if not is_debug else leaf
                      for leaf in subtree.split(" ")]
            yield leaves


@timeit
def index_sent_phrase_no(retagged_sents, scored_sents, is_debug=0, log_time=0):
    # original_word, pos_tag, named entity, tfidf, phrase_index, sentence_index
    info_sents = []
    start = -1
    for i, (retagged_sent, scored_sent) in enumerate(zip(retagged_sents, scored_sents)):
        NPChunker = nltk.RegexpParser(pattern)
        result = Tree.fromstring(str(NPChunker.parse(retagged_sent)))
        chunked = list(traverse_tree(result, 2, is_debug=0))
        phrase_mark = list(chain.from_iterable(
            [len(c)*[j+start+1]
             for j, c in enumerate(chunked)]))
        info_sent = list(map(
            lambda pm_sm_ss: pm_sm_ss[2][1:] + (pm_sm_ss[0], pm_sm_ss[1]),
            zip(phrase_mark, [i]*len(phrase_mark), scored_sent)))
        info_sents.append(info_sent)
        if is_debug:
            # debug_line('orig retagged_sent', retagged_sent)
            # debug_line('chunked', str(chunked), 'red')
            # debug_line('scored sent', scored_sent)
            debug_line('indexed and pos words removed', str(info_sent))
        assert len(phrase_mark) == len(scored_sent), "num of phrase marks %s != num of original items %s\n. \nSomething wrong may happened in retagging. in %sth sentence" % (len(phrase_mark), len(scored_sent), i)
        start = max(phrase_mark)
        if is_debug:
            input('\n\n')
    return info_sents


@timeit
def make_vocab(enc_vocab_counter, dec_vocab_counter, log_time=0):
    print("Writing vocab file...")
    enc_vocab_counter.pop("", "null exists")
    dec_vocab_counter.pop("", "null exists")
    dec_total_words = sum(dec_vocab_counter.values())
    enc_total_words = sum(enc_vocab_counter.values())
    # stats and input vocab size
    dec_most_common = dec_vocab_counter.most_common()
    enc_most_common = enc_vocab_counter.most_common()

    acc_p = 0
    dec_writer = open(join(finished_files_dir, "dec_words"), 'w', 'utf-8')
    enc_writer = open(join(finished_files_dir, "enc_words"), 'w', 'utf-8')

    for word, count in dec_most_common:
        acc_p += (count / dec_total_words)
        dec_writer.write(word + ' ' + str(count) + " " + str(acc_p) + '\n')
    dec_writer.close()
    print("Finished writing dec_vocab file")

    acc_p = 0
    for word, count in enc_most_common:
        acc_p += (count / enc_total_words)
        enc_writer.write(word + ' ' + str(count) + " " + str(acc_p) + '\n')
    enc_writer.close()
    print("Finished writing enc_vocab file")


def list_duplicates(seq):
    seen = set()
    seen_add = seen.add
    # adds all elements it doesn't know yet to seen and all other to seen_twice
    seen_twice = set(x for x in seq if x in seen or seen_add(x))
    # turn the set into a list (as requested)
    return list(seen_twice)


@timeit
def process_vocab(enc_vocab_file, dec_vocab_file, enc_vocab_size, dec_vocab_size, log_time=0):
    missed_words = 0
    missed = open(join(finished_files_dir, 'missedfromenc.txt'), 'w', 'utf-8')
    enc_words = [line.strip().split(" ")[0]
                 for line in open(enc_vocab_file, "r", "utf-8").readlines()]
    dec_vocab = [line.strip().split(' ')[0]
                 for line in open(dec_vocab_file, "r", "utf-8").readlines()[:dec_vocab_size-4]]

    assert len(enc_words) >= enc_vocab_size, 'enc vocab should less than enc words'
    assert len(dec_vocab) == dec_vocab_size-4, 'dec vocab should less than dec words'

    for dec_key in dec_vocab:
        try:
            enc_words.remove(dec_key)
        except:
            missed_words += 1
            missed.write(dec_key+"\n")
            pass

    dec_vocab.extend(dec_must_include)
    assert len(dec_vocab) == len(set(dec_vocab)), "duplicates in dec vocab %s" % str(list_duplicates(dec_vocab))
    dec_writer = open(join(finished_files_dir, "dec_vocab"), 'w', 'utf-8')
    enc_writer = open(join(finished_files_dir, "enc_vocab"), 'w', 'utf-8')
    for w in dec_vocab[:-2]:
        dec_writer.write(w+"\n")
        enc_writer.write(w+"\n")

    for w in dec_vocab[-2:]:
        dec_writer.write(w+"\n")

    enc_vocab_left = enc_vocab_size - dec_vocab_size + 2
    enc_vocab = enc_words[:enc_vocab_left]
    assert len(enc_vocab) == len(set(enc_vocab)), "duplicates in dec vocab %s" % str(list_duplicates(enc_vocab))
    for w in enc_vocab:
        enc_writer.write(w+"\n")

    dec_writer.close()
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


@timeit
def map_tfidf(tagged_sents, normalized_sents, is_debug=0, log_time=0):
    # http://www.davidsbatista.net/blog/2018/02/28/TfidfVectorizer/
    tfidf.fit(normalized_sents)
    vocab = tfidf.vocabulary_
    scored_sents = [
        [
            j + (tfidf.transform([d]).toarray()[0][vocab[i]] if i in vocab else 0,) for i, j in zip(d, t)
        ] for d, t in zip(normalized_sents, tagged_sents)
    ]
    if is_debug:
        for ts in scored_sents:
            debug_line("tfidf scores", str(ts))
            input('\n')

    return scored_sents


def lemma_and_stem(tagged_sent):
    wnled = [port.stem(wnl.lemmatize(w, pos='a' if p[0].lower() == 'j' else p[0].lower()))
             if p[0].lower() in ['j', 'r', 'n', 'v'] and p not in ["NNPS", "NNP"] and w not in stopwords else w
             for w, p in list(map(lambda x: (x[1], x[2]), tagged_sent))]
    assert len(tagged_sent) == len(wnled), "word_normalize: tagged length %s should be equal of standarded len %s\n%s\n%s" % (len(tagged_sent), len(wnled), str(tagged_sent), str(wnled))
    return wnled


@timeit
def word_normalize(tagged_sents, is_debug=False, log_time=0):
    normalized_sents = []
    for tagged_sent in tagged_sents:
        # ported = [port.stem(i) for i in list(map(lambda x: x[0], tagged_sent))]
        # lsed = [ls.stem(i) for i in list(map(lambda x: x[0], tagged_sent))]
        wnled = [port.stem(wnl.lemmatize(w, pos='a' if p[0].lower() == 'j' else p[0].lower()))
                 if p[0].lower() in ['j', 'r', 'n', 'v'] and p not in ["NNPS", "NNP"] and w not in stopwords else w
                 for w, p in list(map(lambda x: (x[1], x[2]), tagged_sent))]
        assert len(tagged_sent) == len(wnled), "word_normalize: tagged length %s should be equal of standarded len %s\n%s\n%s" % (len(tagged_sent), len(wnled), str(tagged_sent), str(wnled))
        normalized_sents.append(wnled)

        if is_debug:
            debug_line("tagged sent", str(tagged_sent))
            # debug_line("sent_stem ported", str(ported))
            # debug_line("sent_stem lancester", str(lsed))
            debug_line("sent_lemma wnled", str(wnled))
            input('\n')
    return normalized_sents


def tagging(tokenized_sents, tag_type='corenlp', is_debug=0, log_time=0):
    if tag_type == "corenlp":
        sents_pos = [pos_tagger.tag(sent) for sent in tokenized_sents]
        # TODO: only a limited ner classes to speed up the process
        sents_ner = [ner_tagger.tag(sent) for sent in tokenized_sents]
    elif tag_type == 'spacy':
        pass

    return sents_pos, sents_ner


@timeit
def tokenize_add_prio(sents, is_debug=False, log_time=0):

    tagged_sents = []
    # these three cost most of the time
    tokenized_sents = [list(tokenize(sent)) for sent in sents]
    sents_pos, sents_ner = tagging(tokenized_sents, tag_type="corenlp", is_debug=0, log_time=1)

    if is_debug:
        for sent in tokenized_sents:
            if len(sent) > 70:
                debug_line('len longer than 70', " ".join(sent), 'red')

    for sp, sn, st in zip(sents_pos, sents_ner, tokenized_sents):
        assert len(sp) == len(sn), "tokenize_add_prio: pos and ner length should be the same, but %s and %s, \n%s\n%s" % (str(sp), str(sn), len(sn), len(sn))
        mapped = []
        case = 1
        for p, n, t in zip(sp, sn, st):
            if case and t.isalpha() and (p[1] not in ['NNP', 'NNPS'] or n[1] == "O"):
                mapped.append((p[0], t.lower(), p[1], n[1]))
                case = 0
            elif case and (p[1] in ['NNP', 'NNPS'] or n[1] != "O"):
                mapped.append((p[0], t, p[1], n[1]))
                case = 0
            else:
                mapped.append((p[0], t, p[1], n[1]))

        # original word, pos_tag, ner_tag
        tagged_sents.append(mapped)
        if is_debug:
            debug_line("sent pos", str(sp))
            debug_line("sent ner", str(sn))
            debug_line("sent combined", str(mapped))
            input('\n')
    return sents_pos, tagged_sents


@timeit
def process_title(title, uppercased, is_debug=False, log_time=0):
    title = title.replace("''", '"')
    tokenized_title = list(tokenize(title))
    if is_debug:
        debug_line("tokenized title", tokenized_title)
        debug_line("uppercased", uppercased)

    title_pos = pos_tagger.tag(tokenized_title)
    if is_debug:
        debug_line("pos tagged title", str(title_pos))
    title_ner = ner_tagger.tag(tokenized_title)
    if is_debug:
        debug_line("ner tagged title", str(title_ner))
    lowercased = []
    for p, n, t in zip(title_pos, title_ner, tokenized_title):
        if t[0].isupper() and t.isalpha() and (p[1] in ['NNP', 'NNPS'] or n[1] != "O"):
            if t in uppercased:
                if is_debug:
                    debug_line("upcase", t)
                lowercased.append(t)
            else:
                if is_debug:
                    debug_line("lowered", t)
                lowercased.append(t.lower())
        else:
            lowercased.append(t.lower())
    if is_debug:
        debug_line("lower cased title", str(lowercased))
    return lowercased


def delete_unk_sents(sents):
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


def sent_filter(sents, debug=False):
    sents = list(filter(lambda x: len(x) > 2, sents))


def load_json(line, is_debug=False):
    _json = json.loads(line, strict=False)
    return _json['id'], _json['title'], _json['content']


def get_pairs_from_corpus(in_file, is_debug=0):
    fi = open(in_file, 'rb')
    lines_count = 0
    illegal_num = 0
    illegal = open(join(finished_files_dir, 'illegal.txt'), 'w', 'utf-8')
    if is_debug:
        frequency = 5
    else:
        frequency = 50000

    while(True):
        line = read_origin(fi, is_debug=0)
        if not line:
            illegal.write("\n\n" + str(illegal_num))
            illegal.close()
            break

        line = bytes2unicode(line)
        _id, title, content = load_json(line)

        if 'u_n_k' in title:
            illegal.write(title + '\n' + content + "\n\n")
            illegal_num += 1
            illegal.flush()
            continue

        sents = cut_sent(content, is_debug=0, log_time=1)
        # sents = delete_unk_sents(sents, is_debug=True)

        # sents = sent_filter(sents, debug=False)

        sents_pos, tagged_sents = tokenize_add_prio(sents, is_debug=0, log_time=1)
        # the tokenized original words are lowercased if needed
        # tagged_sents: pos_tag_word, original word, pos_tag, named entity

        uppercased = set(filter(
            lambda x: x[0].isupper(),
            map(
                lambda x: x[1], chain.from_iterable(tagged_sents)
            )))
        title = process_title(title, list(uppercased), is_debug=0, log_time=1)

        # normalized words only for tfidf scores
        normalized_sents = word_normalize(tagged_sents, is_debug=0, log_time=1)
        # lemmatized and stemmed word, all lowered, stopwords
        # are kept

        # debug_line('the origin sent changed?', str(tagged_sents))
        scored_sents = map_tfidf(tagged_sents, normalized_sents, is_debug=0, log_time=1)
        # pos_tag_word, original word, pos_tag, named entity, tfidf

        ner_pos_tagged_sents = pos_repos_tag(tagged_sents, is_debug=0, log_time=1)
        # repos_tag_word, repos_tag

        indexed_sents = index_sent_phrase_no(ner_pos_tagged_sents, scored_sents, is_debug=0, log_time=1)
        # original_word, pos_tag, named entity, tfidf, phrase_index, sentence_index

        infor_content = list(chain.from_iterable(indexed_sents))

        if is_debug:
            debug_line("title", str(title))
            debug_line("infor_content", str(infor_content))
            input('\n\n')
        lines_count += 1
        if lines_count % frequency == 0:
            print(lines_count)
            print(str(title))
        yield title, infor_content


@timeit
def write_to_text(in_file, out_file, max_length=10*5, makevocab=True, is_debug=0, log_time=0):
    start = time.time()
    sample_count = 0
    out_file = join(finished_files_dir, out_file)
    print('parse corpus from %s to %s' % (in_file, out_file))
    if makevocab:
        enc_vocab_counter = collections.Counter()
        dec_vocab_counter = collections.Counter()

    file_num = 0
    length = 0

    writer = open(out_file + "_" + str(file_num), 'w', 'utf-8')

    for title, infor_content in get_pairs_from_corpus(in_file, is_debug=0):
        sample_count += 1
        if is_debug:
            debug_line('title written', str(title))
        original_words, pos_tags, ner_tags, tfidf_scores, phrase_indices, sent_indices = zip(*infor_content)
        if makevocab:
            enc_vocab_counter.update(original_words)
            dec_vocab_counter.update(title)
        if length >= max_length:
            file_num += 1
            writer.close()
            writer = open(out_file + "_" + str(file_num), 'w', 'utf-8')
            length = 0
        line = "\t".join(
            [" ".join(title),
             " ".join(original_words),
             " ".join(pos_tags),
             " ".join(ner_tags),
             " ".join(map(str, tfidf_scores)),
             " ".join(map(str, phrase_indices)),
             " ".join(map(str, sent_indices))
             ]) + "\n"
        if is_debug:
            debug_line('line written', line)
        writer.write(line)
        length += 1

    writer.close()
    if makevocab:
        # write vocab to file
        make_vocab(enc_vocab_counter, dec_vocab_counter, log_time=1)

    spent = time.time() - start
    print('finished, %s munites spent, %s seconds per sample' % (str(spent/60), spent/sample_count))


if __name__ == '__main__':
    in_file = './data/dptest.txt'
    write_to_text(in_file, "temp", is_debug=0, log_time=1)
