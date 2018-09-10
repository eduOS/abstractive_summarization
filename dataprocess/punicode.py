# -*  coding: utf-8 -*-
# tested only for python3.5

from __future__ import unicode_literals, print_function
from __future__ import absolute_import
from __future__ import division

# from codecs import open
import collections
import json
from utils import read_origin, bytes2unicode, prep_cut_sent, load_json, colored
from codecs import open
import os
from os.path import join
import re
import time
from threading import Thread
from itertools import chain
from utils import debug_line
import pymongo
from six.moves import queue as Queue

import nltk
from utils import timeit
from nltk.tree import Tree
from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.stem import LancasterStemmer
from nltk.corpus import stopwords
from nltk.parse.corenlp import CoreNLPParser

# for spacy
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import multiprocessing
ENC_VOCAB_SIZE = 500000
DEC_VOCAB_SIZE = 100000
NUM_WORKERS = multiprocessing.cpu_count()
# NUM_WORKERS = 1
queue = Queue.Queue()


END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', ")"]

finished_files_dir = "./finished_files/"
if not os.path.exists(finished_files_dir):
    os.makedirs(finished_files_dir)

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


def pos_repos_tag(tagged_sents, is_debug=0, log_time=0):
    """
    pos retag according to the ner tag, the named entity are tagged as <NE>
    """

    pos_retagged_sents = []
    for tagged_sent in tagged_sents:
        length = 0
        current_sent = []
        continuous_chunk = []

        for (pos_token, p_tag, n_tag) in tagged_sent:
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
                leaves = subtree.leaves()
                yield list(chain.from_iterable(list(map(lambda x: x[0].split(" "), leaves))))
                traverse_tree(subtree)
        else:
            # the named entity should be separated
            leaves = subtree[0].split(" ")
            yield leaves


def index_sent_phrase_no(retagged_sents, scored_sents, NPChunker, is_debug=0, log_time=0):
    # original_word, pos_tag, named entity, tfidf, phrase_index, sentence_index
    info_sents = []
    start = -1
    for i, (retagged_sent, scored_sent) in enumerate(zip(retagged_sents, scored_sents)):
        try:
            result = NPChunker.parse(retagged_sent)
        except Exception as e:
            print(str(retagged_sent))
            print()
            print(str(result))
            print()
            print(e)
            continue
        chunked = list(traverse_tree(result, 2, is_debug=0))
        phrase_mark = list(chain.from_iterable(
            [len(c)*[j+start+1]
             for j, c in enumerate(chunked)]))
        info_sent = list(map(
            lambda pm_sm_ss: pm_sm_ss[2] + (pm_sm_ss[0], pm_sm_ss[1]),
            zip(phrase_mark, [i]*len(phrase_mark), scored_sent)))
        info_sents.append(info_sent)
        if is_debug:
            debug_line('phrase mark', phrase_mark)
            debug_line('orig retagged_sent', retagged_sent)
            debug_line('chunked', chunked, 'red')
            debug_line('scored sent', scored_sent)
            debug_line('indexed', info_sent)
        assert len(phrase_mark) == len(scored_sent), "num of phrase marks %s\n%s != num of original items %s\n%s\n. \
            The retagged sent is %s; \nSomething wrong may happened in retagging. in %sth sentence" % (
            len(phrase_mark), str(phrase_mark), len(scored_sent), str(scored_sent), str(retagged_sent), i)
        start = max(phrase_mark)
        if is_debug:
            input('\n\n')
    return info_sents


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


def map_tfidf(tagged_sents, normalized_sents, tfidf, is_debug=0, log_time=0):
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


def word_normalize(tagged_sents, port, wnl, is_debug=False, log_time=0):
    normalized_sents = []
    for tagged_sent in tagged_sents:
        # ported = [port.stem(i) for i in list(map(lambda x: x[0], tagged_sent))]
        # lsed = [ls.stem(i) for i in list(map(lambda x: x[0], tagged_sent))]
        wnled = [port.stem(wnl.lemmatize(w, pos='a' if p[0].lower() == 'j' else p[0].lower()))
                 if p[0].lower() in ['j', 'r', 'n', 'v'] and w not in stopwords else w
                 for w, p in list(map(lambda x: (x[0], x[1]), tagged_sent))]
        assert len(tagged_sent) == len(wnled), "word_normalize: tagged length %s should be equal of standarded len %s\n%s\n%s" % (len(tagged_sent), len(wnled), str(tagged_sent), str(wnled))
        normalized_sents.append(wnled)

        if is_debug:
            debug_line("tagged sent", str(tagged_sent))
            # debug_line("sent_stem ported", str(ported))
            # debug_line("sent_stem lancester", str(lsed))
            debug_line("sent_lemma wnled", str(wnled))
            input('\n')
    return normalized_sents


def tokenize_add_prio(sents, tokenize, pos_tagger, ner_tagger, is_debug=False, log_time=0):

    tagged_sents = []
    # these three cost most of the time

    tokenized_sents = [list(tokenize(sent)) for sent in sents]
    sents_pos = [pos_tagger.tag(sent) for sent in tokenized_sents]
    # TODO: only a limited ner classes to speed up the process
    sents_ner = [ner_tagger.tag(sent) for sent in tokenized_sents]

    if is_debug:
        for sent in tokenized_sents:
            if len(sent) > 70:
                debug_line('len longer than 70', " ".join(sent), 'red')

    for sp, sn, st in zip(sents_pos, sents_ner, tokenized_sents):
        assert len(sp) == len(sn), (
            "tokenize_add_prio: pos, ner and tokenized words length should be the same, but %s, %s, \n%s\n%s" % (len(sp), len(sn), str(sp), str(sn))
        )
        mapped = []
        for p, n, t in zip(sp, sn, st):
            mapped.append((p[0].lower(), p[1], n[1]))

        # original word, pos_tag, ner_tag
        tagged_sents.append(mapped)
        if is_debug:
            debug_line("sent pos", str(sp))
            debug_line("sent ner", str(sn))
            debug_line("sent combined", str(mapped))
            input('\n')
    return sents_pos, tagged_sents


def process_title(title, tokenize, pos_tagger, ner_tagger, uppercased, is_debug=False, log_time=0):
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

    lowercased = list(map(lambda x: x.lower(), tokenized_title))
    if is_debug:
        debug_line("lower cased title", str(lowercased))
    return lowercased


def delete_unk_sents(sents):
    """
    delete by hand, if too many unks
    """
    return


def sent_filter(sents, debug=False):
    sents = list(filter(lambda x: len(x) > 2, sents))


@timeit
def process_one_sample(generator, mydb, tokenize, pos_tagger, ner_tagger, port, wnl, tfidf, NPChunker, log_time=0):
    try:
        triple = generator.next()
    except TypeError:
        print(colored('None object and break', 'red'))
        return False
    _id, title, content = triple['id_'], triple['orig_title'], triple['content_sents']
    # sents = prep_cut_sent(content, is_debug=0, log_time=0)
    sents = [sent.strip() for sent in content.split('\n') if len(sent.strip()) > 1]

    # sents = delete_unk_sents(sents, is_debug=True)
    # sents = sent_filter(sents, debug=False)

    # the tokenized original words are lowercased if needed
    sents_pos, tagged_sents = tokenize_add_prio(sents, tokenize, pos_tagger, ner_tagger, is_debug=0, log_time=1)
    # tagged_sents: pos_tag_word, original word, pos_tag, named entity

    uppercased = set(filter(
        lambda x: x[0].isupper(),
        map(
            lambda x: x[1], chain.from_iterable(tagged_sents)
        )))
    title = process_title(title, tokenize, pos_tagger, ner_tagger, list(uppercased), is_debug=0, log_time=1)

    # normalized words only for tfidf scores
    normalized_sents = word_normalize(tagged_sents, port, wnl, is_debug=0, log_time=1)
    # lemmatized and stemmed word, all lowered, stopwords
    # are kept

    # debug_line('the origin sent changed?', str(tagged_sents))
    scored_sents = map_tfidf(tagged_sents, normalized_sents, tfidf, is_debug=0, log_time=1)
    # pos_tag_word, original word, pos_tag, named entity, tfidf

    ner_pos_tagged_sents = pos_repos_tag(tagged_sents, is_debug=0, log_time=1)
    # repos_tag_word, repos_tag

    indexed_sents = index_sent_phrase_no(ner_pos_tagged_sents, scored_sents, NPChunker, is_debug=0, log_time=1)
    # original_word, pos_tag, named entity, tfidf, phrase_index, sentence_index

    infor_content = list(chain.from_iterable(indexed_sents))
    original_words, pos_tags, ner_tags, tfidf_scores, phrase_indices, sent_indices = zip(*infor_content)
    mydb.bytecup_2018.update_many(
        {"_id": _id},
        {
            "$set": {
                "original_words": original_words,
                "pos_tags": pos_tags,
                "ner_tags": ner_tags,
                "tfidf_scores": tfidf_scores,
                "phrase_indices": phrase_indices,
                "sent_indices": sent_indices,
            }
        }, upsert=False)
    return True


def processor(st=0, is_debug=0, log_time=0):
    tokenize = CoreNLPParser(url='http://localhost:9000').tokenize
    pos_tagger = CoreNLPParser(url='http://localhost:9000', tagtype='pos')
    ner_tagger = CoreNLPParser(url='http://localhost:9000', tagtype='ner')
    tfidf = TfidfVectorizer(stop_words=stopwords, analyzer='word', tokenizer=dummy_fun, preprocessor=dummy_fun, token_pattern=None)
    NPChunker = nltk.RegexpParser(pattern)

    port = PorterStemmer()
    wnl = WordNetLemmatizer()
    # ls = LancasterStemmer()

    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    mydb = myclient["mydatabase"]
    generator = mydb.bytecup_2018.find({"id_": {"$gt": st, "$lt": st+50000}})

    while(True):
        state = process_one_sample(generator, mydb, tokenize, pos_tagger, ner_tagger, port, wnl, tfidf, NPChunker, log_time=2)
        if not state:
            break


def multi_process_corpus(is_debug=0):
    channels = list(range(0, 10**6, 5000))[:NUM_WORKERS]
    wn.ensure_loaded()
    threads = []
    for i in channels:
        new_t = Thread(target=processor, kwargs={"st": i})
        threads.append(new_t)
        threads[-1].start()
        print(str(threads[-1]) + ' starts')


@timeit
def write_to_text(in_file, out_file, max_length=10*5, makevocab=True, is_debug=0, log_time=0):
    start = time.time()
    sample_count = 0
    out_file = join(finished_files_dir, out_file)
    print('parse corpus from %s to %s' % (in_file, out_file))
    file_num = 0
    length = 0

    writer = open(out_file + "_" + str(file_num), 'w', 'utf-8')

    for title, infor_content in process_pairs():
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
    in_file = './data/bytecup.corpus.train.txt'
    multi_process_corpus(make_vocab)
    # write_to_text(in_file, "temp", is_debug=0, log_time=0)
