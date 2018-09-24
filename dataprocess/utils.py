# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function
from __future__ import absolute_import
from __future__ import division
from termcolor import colored
from itertools import chain

# from cntk.tokenizer import JiebaTokenizer
# from cntk.cleanser import Cleanser
# from cntk.standardizer import Standardizer
from nltk.tree import Tree
import time
import re
import unicodedata
import json
from unidecode import unidecode

# tokenizer = JiebaTokenizer()
# standardizer = Standardizer()
# cleanser = Cleanser()

# punc_kept = u" ；;。!！,：(（）:?)《》、？，%~"


# def sourceline2words(line, with_digits=True):
#     if with_digits:
#         line = standardizer.set_sentence(line).fwidth2hwidth().to_lowercase().sentence
#     else:
#         line = standardizer.set_sentence(line).fwidth2hwidth().digits().to_lowercase().sentence
#     words = tokenizer.sentence2words(line)
#     line = " ".join(words)
#     line = cleanser.set_sentence(line).delete_useless().sentence
#     words = [w for w in line.split() if w]
#     return words


t_time = time.time()
count_s = 0


def timeit(method):
    def timed(*args, **kw):
        global count_s
        ts = time.time()
        result = method(*args, **kw)
        count_s += 1
        te = time.time()
        if 'log_time' in kw and kw['log_time'] == 2 and count_s % 200 == 0:
            speed = count_s/(te-t_time)
            print(colored('%r  %2.2f samples per second on everage, time left: %2.2f hours' % (method.__name__, speed, (int(702484/4) - count_s)/(speed * 3600)), 'green'))

        elif 'log_time' in kw and kw['log_time'] == 1:
            if te - ts > 1:
                print(colored('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000), 'red'))
            else:
                print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result
    return timed


def read_origin(fi, is_debug=False):
    line = fi.readline()
    if not line:
        return None
    if is_debug:
        debug_line('origin', line)
    return line


def debug_line(tag, line, color="green"):
    leng = str(len(line))
    line = str(line)
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
        print(line + "\t" + leng)
    elif not ipt:
        print(line[:default_len] + "\t" + leng, end="")
        print("...")
    elif ipt:
        print(line[:int(ipt)] + "\t" + leng, end="")
        print("...")


def load_json(line, is_debug=False):
    _json = json.loads(line, strict=False)
    try:
        title = _json['title']
    except:
        title = None
    return _json['id'], title, _json['content']


def prep_cut_sent(line, max_sent_words=60, max_sents=70, is_debug=False, log_time=0):
    """
    TODO: combining the existing tools may be better
    """
    if is_debug:
        debug_line("have quote", line)
    line = line.replace("''", '"')
    if is_debug:
        debug_line("replace quote", line)

    sents = re.sub(r'''((?<![A-Z])\.(?=[A-Z][^.])|\.(?=([ '"]+|by))\)*|((?<!\d):|[;!?])['"]*)|(?<![\d ])\.(?=\d\.?)''', r'\1\n', line).split('\n')
    if is_debug:
        for sent in sents:
            debug_line("cut sent", sent)

    new_sents = []
    for sent in sents:
        sent = sent.strip()
        if len(sent) < 1:
            continue
        sp_sent = sent.split()
        if sp_sent.count('u_n_k') > len(sp_sent) / 2:
            continue
        if len(sp_sent) > max_sent_words:
            if is_debug:
                debug_line("large sent", sent, 'red')
                input()
            # if
            sub_sents = re.sub(r"(?<=[^,]{120}),", r'\n', sent).split('\n')
            if is_debug:
                debug_line("recut sent", str(sub_sents))
                input()
            for sub_sent in sub_sents:
                sub_sent_len = len(sub_sent.split())
                if sub_sent_len < max_sent_words and sub_sent_len > 1:
                    if is_debug:
                        debug_line("good recut sent", str(sub_sent))
                        input()
                    new_sents.append(sub_sent)
                    if len(new_sents) > max_sents:
                        break
                else:
                    if is_debug:
                        debug_line("bad recut sent", str(sub_sents), 'red')
                        input()
        elif len(sent) > 1:
            if is_debug:
                debug_line("good sent", sent)
            new_sents.append(sent)
            if len(new_sents) > max_sents:
                break

    if is_debug:
        for sent in new_sents:
            debug_line("my cut sent", sent)
        input('\n\n')
        return new_sents
    else:
        return new_sents


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



def process_title(title, tokenize, pos_tagger, dec_queue=None, makevocab=0, is_debug=False, log_time=0, max_len=18):
    title = title.replace("''", '"')
    tokenized_title = list(tokenize(title))
    if is_debug:
        debug_line("tokenized title", tokenized_title)

    title_pos = pos_tagger.tag(tokenized_title)

    lowercased = list(map(lambda x: x[0].lower(), title_pos))
    if makevocab:
        dec_queue.put(dict(Counter(lowercased)))

    if is_debug:
        debug_line("lower cased title", str(lowercased))
    lowercased = lowercased[:max_len]
    return lowercased
