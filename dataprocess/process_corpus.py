# -*  coding: utf-8 -*-
# tested only for python3.5

from __future__ import unicode_literals, print_function
from __future__ import absolute_import
from __future__ import division

# from codecs import open
from collections import Counter
# import json
from utils import colored
from codecs import open
import os
# from os.path import join
# import re
import time
import pickle
from threading import Thread
from itertools import chain
from utils import debug_line
import pymongo
from six.moves import queue as Queue

import nltk
from utils import timeit
from utils import tokenize_add_prio
from utils import pos_repos_tag
from utils import traverse_tree
from utils import process_title
from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer, WordNetLemmatizer
# from nltk.stem import LancasterStemmer
from nltk.corpus import stopwords
from nltk.parse.corenlp import CoreNLPParser
from pymongo.errors import CursorNotFound

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
stem_counter = Counter()
# pos_tag_vocab = []
# ner_tag_vocab = []
dec_vocab_counter = Counter()


END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', ")"]

finished_files_dir = "./finished_files/"
if not os.path.exists(finished_files_dir):
    os.makedirs(finished_files_dir)

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

stopwords = stopwords.words('english') + list(string.punctuation) + ["-lrb-", '-rrb-', 'u_n_k']


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
    print("counting words frequence ...")
    l = 0
    while(l < 10):
        case = 0
        if not enc_queue.empty():
            e_words = enc_queue.get()
            enc_vocab_counter.update(e_words[0])
            stem_counter.update(e_words[1])
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
def make_whole_vocab(machine_num, log_time=0, _continue=0):
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    mydb = myclient["mydatabase"]
    mycol = mydb["bytecup2018"]
    vocab_name = "vocab_freq_dict_" + str(machine_num)
    if not _continue and list(mycol.find({"_id": vocab_name})):
        try:
            mycol.remove({"_id": vocab_name})
        except:
            try:
                mycol.delete_one({"_id": vocab_name})
            except:
                pass

    try:
        if not _continue:
            enc_dict = dict(enc_vocab_counter)
            dec_dict = dict(dec_vocab_counter)
            stem_dict = dict(stem_counter)
            enc_vocab = sorted(list(zip(enc_dict.keys(), enc_dict.values())), key=lambda x: x[1], reverse=True)
            dec_vocab = sorted(list(zip(dec_dict.keys(), dec_dict.values())), key=lambda x: x[1], reverse=True)
            stem_vocab = sorted(list(zip(stem_dict.keys(), stem_dict.values())), key=lambda x: x[1], reverse=True)

            mycol.insert_one(
                {
                    "_id": vocab_name,
                    "enc_vocab_freq_dict": enc_vocab,
                    "dec_vocab_freq_dict": dec_vocab,
                    "stem_vocab_freq_dict": stem_vocab,
                }
            )
        else:
            v = mycol.find({"_id": vocab_name}).next()
            enc_vocab_counter.update(dict(v["enc_vocab_freq_dict"]))
            dec_vocab_counter.update(dict(v["dec_vocab_freq_dict"]))
            stem_counter.update(dict(v["stem_vocab_freq_dict"]))
            enc_dict = dict(enc_vocab_counter)
            dec_dict = dict(dec_vocab_counter)
            stem_dict = dict(stem_counter)
            enc_vocab = sorted(list(zip(enc_dict.keys(), enc_dict.values())), key=lambda x: x[1], reverse=True)
            dec_vocab = sorted(list(zip(dec_dict.keys(), dec_dict.values())), key=lambda x: x[1], reverse=True)
            stem_vocab = sorted(list(zip(stem_dict.keys(), stem_dict.values())), key=lambda x: x[1], reverse=True)
            try:
                mycol.remove({"_id": vocab_name})
            except:
                try:
                    mycol.delete_one({"_id": vocab_name})
                except:
                    pass

            mycol.insert_one(
                {
                    "_id": vocab_name,
                    "enc_vocab_freq_dict": enc_vocab,
                    "dec_vocab_freq_dict": dec_vocab,
                    "stem_vocab_freq_dict": stem_vocab,
                }
            )
    except Exception as e:
        enc_dict = dict(enc_vocab_counter)
        dec_dict = dict(dec_vocab_counter)
        stem_dict = dict(stem_counter)
        enc_vocab = sorted(list(zip(enc_dict.keys(), enc_dict.values())), key=lambda x: x[1], reverse=True)
        dec_vocab = sorted(list(zip(dec_dict.keys(), dec_dict.values())), key=lambda x: x[1], reverse=True)
        stem_vocab = sorted(list(zip(stem_dict.keys(), stem_dict.values())), key=lambda x: x[1], reverse=True)

        enc_f = open('enc_vocab_fre_dict.pkl', 'w')
        dec_f = open('dec_vocab_fre_dict.pkl', 'w')
        stem_f = open('stem_vocab_fre_dict.pkl', 'w')
        pickle.dump(enc_vocab, enc_f)
        pickle.dump(dec_vocab, dec_f)
        pickle.dump(stem_vocab, stem_f)
        enc_f.close()
        dec_f.close()
        stem_f.close()
        print(e)
        print('enc and dec vocab freq has been dumped as pickle')


def map_tfidf(tagged_sents, normalized_sents, tfidf, is_debug=0, log_time=0):
    # http://www.davidsbatista.net/blog/2018/02/28/TfidfVectorizer/
    tfidf.fit(normalized_sents)
    vocab = tfidf.vocabulary_
    scored_sents = [
        [
            j + (i, tfidf.transform([d]).toarray()[0][vocab[i]] if i in vocab else 0) for i, j in zip(d, t)
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
            debug_line("sent_stem wnled", str(wnled))
            input('\n')
    return normalized_sents


def delete_unk_sents(sents):
    """
    delete by hand, if too many unks
    """
    return


def sent_filter(sents, debug=False):
    sents = list(filter(lambda x: len(x) > 2, sents))


@timeit
def process_one_sample(generator, mycol, tokenize, pos_tagger, ner_tagger, port, wnl, tfidf, NPChunker, makevocab=0, log_time=0):
    try:
        triple = generator.next()
    except TypeError:
        print(colored('None object and break', 'red'))
        return False
    except CursorNotFound as e:
        print(colored('cursor not found timeout error', 'red'))
        print(e)
        return "error"
    except StopIteration:
        print(colored('stop iteration and break', 'red'))
        return False
    except Exception as e:
        print(e)
        return "error"
    _id, title, content = triple['new_id'], triple['orig_title'], triple['content_sents']
    # sents = prep_cut_sent(content, is_debug=0, log_time=0)
    sents = content.split('\n')

    # sents = delete_unk_sents(sents, is_debug=True)
    # sents = sent_filter(sents, debug=False)

    # the tokenized original words are lowercased if needed
    sents_pos, tagged_sents = tokenize_add_prio(sents, tokenize, pos_tagger, ner_tagger, is_debug=0, log_time=1)
    # tagged_sents: pos_tag_word, pos_tag, named entity

    if title is not None:
        title = process_title(title, tokenize, pos_tagger, dec_queue, makevocab=1, is_debug=0, log_time=1)

    # normalized words only for tfidf scores
    normalized_sents = word_normalize(tagged_sents, port, wnl, is_debug=0, log_time=1)
    # stemtized and stemmed word, all lowered, stopwords
    # are kept

    # debug_line('the origin sent changed?', str(tagged_sents))
    try:
        scored_sents = map_tfidf(tagged_sents, normalized_sents, tfidf, is_debug=0, log_time=1)
    except Exception as e:
        print(_id)
        print(e)
        return "error"
    # pos_tag_word, pos_tag, named entity, stem&lemmed, tfidf

    ner_pos_tagged_sents = pos_repos_tag(tagged_sents, is_debug=0, log_time=1)
    # repos_tag_word, repos_tag

    indexed_sents = index_sent_phrase_no(ner_pos_tagged_sents, scored_sents, NPChunker, is_debug=0, log_time=1)
    # pos_tag, named entity, stem&lemmed, tfidf, phrase_index, sentence_index

    infor_content = list(chain.from_iterable(indexed_sents))
    pos_tag_words, pos_tags, ner_tags, stem, tfidf_scores, phrase_indices, sent_indices = zip(*infor_content)
    if makevocab:
        enc_queue.put(
            (
                dict(Counter(pos_tag_words)),
                dict(Counter(stem))
            )
        )
    mycol.update_many(
        {"new_id": _id},
        {
            "$set": {
                # the title should be added here
                "pos_tag_words": pos_tag_words,
                "pos_tags": pos_tags,
                "ner_tags": ner_tags,
                "stem": stem,
                "tfidf_scores": tfidf_scores,
                "phrase_indices": phrase_indices,
                "sent_indices": sent_indices,
            }
        }, upsert=False)
    return True


def processor(st=0, ed=0, makevocab=0, is_debug=0, log_time=0, _continue=False):
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
    mycol = mydb["bytecup2018"]
    if _continue:
        generator = mycol.find({
            "new_id": {"$gt": st, "$lt": ed},
            "pos_tags": {"$exists": False}
        }, no_cursor_timeout=True)
    else:
        generator = mycol.find({
            "new_id": {"$gt": st, "$lt": ed},
        }, no_cursor_timeout=True)
    count = 0

    while(True):
        state = process_one_sample(generator, mycol, tokenize, pos_tagger, ner_tagger, port, wnl, tfidf, NPChunker, makevocab=makevocab, log_time=2)
        if not state:
            print(count)
            generator.close()
            break
        elif state != "error":
            count += 1


def multi_process_corpus(makevocab=0, is_debug=0):
    # 95:0, 96:1, 160:2, 243:3
    machine_num = 0
    _continue = True
    # https://stackoverflow.com/a/11554877/3552975

    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    mydb = myclient["mydatabase"]
    generator = mydb.bytecup2018.find({})
    total_sample = len(list(map(lambda x: x['_id'], generator)))
    print("the total number of samples is %s" % str(total_sample))
    total_sample = int(total_sample / 4)
    width = int(total_sample/NUM_WORKERS)
    print("%s threads, %s samples for each" % (NUM_WORKERS, width))
    generator.close()

    channels = list(range(total_sample*machine_num, total_sample*(machine_num+1), width))
    wn.ensure_loaded()
    threads = []
    for i, st in enumerate(channels):
        time.sleep(3)
        if i == len(channels)-1:
            width = width + NUM_WORKERS

        new_t = Thread(target=processor, kwargs={"st": st, 'ed': st+width, "makevocab": makevocab, "_continue": _continue})
        threads.append(new_t)
        threads[-1].start()
        print(str(threads[-1]) + ' starts for sample %s-%s' % (st, st+width))

    if makevocab:
        print('start counting words')
        count_words()
        print('start making vocabulary')
        make_whole_vocab(machine_num, _continue=_continue)
        print('vocab make thread starts')


if __name__ == '__main__':
    in_file = './data/bytecup.corpus.train.txt'
    makevocab = 1
    if "train" not in in_file and makevocab:
        raise("not train corpus do not make vocab")

    # if makevocab:
    #     input("are you sure to make vocab?")
    multi_process_corpus(makevocab=makevocab)
    # write_to_text(in_file, "temp", is_debug=0, log_time=0)
    # 702484
