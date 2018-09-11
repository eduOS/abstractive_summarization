# -*  coding: utf-8 -*-
# tested only for python3.5

from __future__ import unicode_literals, print_function
from __future__ import absolute_import
from __future__ import division

# from codecs import open
from collections import Counter
import json
from utils import read_origin, bytes2unicode, prep_cut_sent, load_json, colored
from codecs import open
import os
from os.path import join
import re
import time
from multiprocessing import Process
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
from pymongo.errors import CursorNotFound

# for spacy
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import multiprocessing
ENC_VOCAB_SIZE = 500000
DEC_VOCAB_SIZE = 100000
NUM_WORKERS = int(multiprocessing.cpu_count() / 2)
# NUM_WORKERS = 1
enc_queue = Queue.Queue()
dec_queue = Queue.Queue()
enc_vocab_counter = Counter()
dec_vocab_counter = Counter()


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


def count_words(log_time=0):
    print("Writing vocab file...")
    l = 0
    while(l < 10):
        case = 0
        if not enc_queue.empty():
            e_words = enc_queue.get()
            enc_vocab_counter.update(e_words)
            l = 0
            time.sleep(0.5)
        else:
            case += 1
        if not dec_queue.empty():
            d_words = dec_queue.get()
            dec_vocab_counter.update(d_words)
            l = 0
            time.sleep(0.5)
        else:
            case += 1

        if case == 2:
            l += 1
            if l > 5:
                print('sleeping %s seconds.. length of enc counter %s' % (str(l), str(len(enc_vocab_counter))))
            time.sleep(l)


def list_duplicates(seq):
    seen = set()
    seen_add = seen.add
    # adds all elements it doesn't know yet to seen and all other to seen_twice
    seen_twice = set(x for x in seq if x in seen or seen_add(x))
    # turn the set into a list (as requested)
    return list(seen_twice)


@timeit
def make_vocab(log_time=0):
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    mydb = myclient["mydatabase"]
    enc_dict = dict(enc_vocab_counter)
    dec_dict = dict(dec_vocab_counter)
    mydb.bytecup_2018.insert_one(
        {
            "_id": "vocab_freq_dict",
            "enc_vocab_dict": list(zip(enc_dict.keys(), enc_dict.values())),
            "dec_vocab_dict": list(zip(dec_dict.keys(), dec_dict.values())),
        }
    )


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


def process_title(title, tokenize, pos_tagger, ner_tagger, uppercased, makevocab=0, is_debug=False, log_time=0):
    title = title.replace("''", '"')
    tokenized_title = list(tokenize(title))
    if is_debug:
        debug_line("tokenized title", tokenized_title)
        debug_line("uppercased", uppercased)

    title_pos = pos_tagger.tag(tokenized_title)

    lowercased = list(map(lambda x: x[0].lower(), title_pos))
    if makevocab:
        dec_queue.put(dict(Counter(lowercased)))

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
def process_one_sample(generator, mydb, tokenize, pos_tagger, ner_tagger, port, wnl, tfidf, NPChunker, makevocab=0, log_time=0):
    try:
        triple = generator.next()
    except TypeError:
        print(colored('None object and break', 'red'))
        return False
    except CursorNotFound:
        print(colored('cursor not found timeout error', 'red'))
        return False
    _id, title, content = triple['new_id'], triple['orig_title'], triple['content_sents']
    # sents = prep_cut_sent(content, is_debug=0, log_time=0)
    sents = content.split('\n')

    # sents = delete_unk_sents(sents, is_debug=True)
    # sents = sent_filter(sents, debug=False)

    # the tokenized original words are lowercased if needed
    sents_pos, tagged_sents = tokenize_add_prio(sents, tokenize, pos_tagger, ner_tagger, is_debug=0, log_time=1)
    # tagged_sents: pos_tag_word, pos_tag, named entity

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
    # pos_tag_word, pos_tag, named entity, tfidf

    ner_pos_tagged_sents = pos_repos_tag(tagged_sents, is_debug=0, log_time=1)
    # repos_tag_word, repos_tag

    indexed_sents = index_sent_phrase_no(ner_pos_tagged_sents, scored_sents, NPChunker, is_debug=0, log_time=1)
    # pos_tag, named entity, tfidf, phrase_index, sentence_index

    infor_content = list(chain.from_iterable(indexed_sents))
    pos_tag_words, pos_tags, ner_tags, tfidf_scores, phrase_indices, sent_indices = zip(*infor_content)
    if makevocab:
        enc_queue.put(dict(Counter(pos_tag_words)))
    mydb.bytecup_2018.update_many(
        {"new_id": _id},
        {
            "$set": {
                "original_words": pos_tag_words,
                "pos_tags": pos_tags,
                "ner_tags": ner_tags,
                "tfidf_scores": tfidf_scores,
                "phrase_indices": phrase_indices,
                "sent_indices": sent_indices,
            }
        }, upsert=False)
    return True


def processor(st=0, ed=0, makevocab=0, is_debug=0, log_time=0):
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
    generator = mydb.bytecup_2018.find({"new_id": {"$gt": st, "$lt": ed}}, no_cursor_timeout=True)
    count = 0

    while(True):
        state = process_one_sample(generator, mydb, tokenize, pos_tagger, ner_tagger, port, wnl, tfidf, NPChunker, makevocab=makevocab, log_time=2)
        if not state:
            print(count)
            generator.close()
            break
        else:
            count += 1


def multi_process_corpus(makevocab=0, is_debug=0):
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    mydb = myclient["mydatabase"]
    generator = mydb.bytecup_2018.find({})
    total_sample = len(list(map(lambda x: x['_id'], generator)))
    print("the total number of samples is %s" % str(total_sample))
    total_sample = int(total_sample / 10)
    width = int(total_sample/NUM_WORKERS)
    print("%s threads, %s samples for each" % (NUM_WORKERS, width))

    channels = list(range(0, total_sample, width))
    wn.ensure_loaded()
    threads = []
    for i, st in enumerate(channels):
        time.sleep(3)
        if i == len(channels)-1:
            width = width + NUM_WORKERS
        new_t = Thread(target=processor, kwargs={"st": st, 'ed': st+width, "makevocab": makevocab})
        threads.append(new_t)
        threads[-1].start()
        print(str(threads[-1]) + ' starts(%s-%s)' % (st, st+width))
    print('start counting words')
    count_words()
    print('start making vocabulary')
    make_vocab()
    print('vocab make thread starts')


if __name__ == '__main__':
    in_file = './data/bytecup.corpus.train.txt'
    multi_process_corpus(makevocab=1)
    # write_to_text(in_file, "temp", is_debug=0, log_time=0)
    # 702484
