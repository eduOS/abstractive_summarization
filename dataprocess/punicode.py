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
from itertools import chain
import nltk
from nltk.tree import Tree
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.stem import LancasterStemmer
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import TfidfVectorizer
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

pattern = r"""
NP: {<DT>?<JJ.?>*(<NN.?>|<NE>)}
VBD: {<VBD>}
IN: {<IN>}
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


def cut_sent(line, is_debug=False):
    """
    TODO: combining the existing tools may be better
    """
    line = line.replace("''", '"')
    sents = re.sub(r'''((?<![A-Z])\.(?=[A-Z][^.])|\.(?=([ '"]+|by))\)*|[;!?:]['"]*)|(?<![\d ])\.(?=\d\.?)''', r'\1\n', line).split('\n')
    sents = [sent.strip() for sent in sents if len(sent.strip()) > 1]

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


def pos_repos_tag(tagged_sents, is_debug=0):
    """
    pos retag according to the ner tag, the named entity are tagged as <NE>
    """

    pos_retagged_sents = []
    for tagged_sent in tagged_sents:
        current_sent = []
        continuous_chunk = []

        for (token, p_tag, n_tag) in tagged_sent:
            if n_tag != "O":
                continuous_chunk.append(token)
            else:
                if continuous_chunk:  # if the current chunk is not empty
                    if is_debug:
                        retagged = ("_".join(continuous_chunk), "_NE_")
                    else:
                        retagged = (" ".join(continuous_chunk), "NE")
                    current_sent.append(retagged)
                    continuous_chunk = []
                current_sent.append((token, p_tag))

        if is_debug:
            debug_line("origial sent", str(tagged_sent))
            debug_line("retagged sent", str(current_sent), 'red')
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
            leaves = [leaf.split('/')[0] if not is_debug else leaf for leaf in subtree.split(" ")]
            yield leaves


def index_sent_phrase_no(retagged_sents, scored_sents, is_debug=0):
    info_sents = []
    start = -1
    for i, (sent, scored_sent) in enumerate(zip(retagged_sents, scored_sents)):
        NPChunker = nltk.RegexpParser(pattern)
        result = Tree.fromstring(str(NPChunker.parse(sent)))
        chunked = list(traverse_tree(result, 2, is_debug=is_debug))
        phrase_mark = list(chain.from_iterable(
            [len(c)*[j+start+1]
             for j, c in enumerate(chunked)]))
        info_sent = list(map(
            lambda pm_sm_ss: pm_sm_ss[2] + ((pm_sm_ss[0], pm_sm_ss[1]),),
            zip(phrase_mark, [i]*len(phrase_mark), scored_sent)))
        info_sents.append(info_sent)
        if is_debug:
            debug_line('orig sent', sent)
            result.pretty_print()
            debug_line('chunked', str(chunked), 'red')
            debug_line('index', str(info_sent))
        assert len(phrase_mark) == len(scored_sent), "num of phrase marks %s != num of original items %s. Something wrong may happened in retagging." % (len(phrase_mark), len(scored_sent))
        start = max(phrase_mark)
        input('\n\n')
    return info_sents


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


def map_tfidf(tagged_sents, normalized_sents, is_debug=True):
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


def word_normalize(tagged_sents, is_debug=False):
    normalized_sents = []
    for tagged_sent in tagged_sents:
        # ported = [port.stem(i) for i in list(map(lambda x: x[0], tagged_sent))]
        # lsed = [ls.stem(i) for i in list(map(lambda x: x[0], tagged_sent))]
        wnled = [port.stem(wnl.lemmatize(w, pos='a' if p[0].lower() == 'j' else p[0].lower()))
                 if p[0].lower() in ['j', 'r', 'n', 'v'] and p not in ["NNPS", "NNP"] else w
                 for w, p in list(map(lambda x: (x[0], x[1]), tagged_sent))]
        normalized_sents.append(wnled)

        if is_debug:
            debug_line("tagged sent", str(tagged_sent))
            # debug_line("sent_stem ported", str(ported))
            # debug_line("sent_stem lancester", str(lsed))
            debug_line("sent_lemma wnled", str(wnled))
            input('\n')
    return normalized_sents


def tokenize_add_prio(sents, is_debug=False):

    tagged_sents = []
    tokenized_sents = [list(tokenize(sent)) for sent in sents]
    sents_pos = [pos_tagger.tag(sent) for sent in tokenized_sents]
    sents_ner = [ner_tagger.tag(sent) for sent in tokenized_sents]
    for sp, sn in zip(sents_pos, sents_ner):
        mapped = list(map(
            lambda i_y: (
                i_y[1][0][0].lower() if i_y[0] < 2 and i_y[1][0][1] not in ['UH', 'NNP', 'NNPS'] else i_y[1][0][0],
                i_y[1][0][1],
                i_y[1][1][1]),
            enumerate(zip(sp, sn))))
        # original word, pos_tag, ner_tag
        tagged_sents.append(mapped)
        if is_debug:
            debug_line("sent pos", str(sp))
            debug_line("sent ner", str(sn))
            debug_line("sent combined", str(mapped))
            input('\n')
    return sents_pos, tagged_sents


def process_title(title, is_debug=False):
    if 'u_n_k' in title:
        return None
    else:
        title = title.replace("''", '"')
        title = tokenize(title)
        title = ' '.join(list(title)).lower()
    return title


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


def main(makevocab=True):
    if makevocab:
        enc_vocab_counter = collections.Counter()
        dec_vocab_counter = collections.Counter()
    fi = open('./data/dptest.txt', 'rb')

    while(True):
        line = read_origin(fi, is_debug=0)
        if not line:
            break

        line = bytes2unicode(line)
        _id, title, content = load_json(line)

        title = process_title(title)
        if not title:
            continue
        sents = cut_sent(content, is_debug=False)
        # sents = delete_unk_sents(sents, is_debug=True)

        # sents = sent_filter(sents, debug=False)
        sents_pos, tagged_sents = tokenize_add_prio(sents, is_debug=0)
        normalized_sents = word_normalize(tagged_sents, is_debug=0)
        # debug_line('the origin sent changed?', str(tagged_sents))
        scored_sents = map_tfidf(tagged_sents, normalized_sents, is_debug=0)
        ner_pos_tagged_sents = pos_repos_tag(tagged_sents, is_debug=0)
        indexed_sents = index_sent_phrase_no(ner_pos_tagged_sents, scored_sents, is_debug=1)

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
