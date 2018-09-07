# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function
from __future__ import absolute_import
from __future__ import division
from termcolor import colored

# from cntk.tokenizer import JiebaTokenizer
# from cntk.cleanser import Cleanser
# from cntk.standardizer import Standardizer
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
        if 'log_time' in kw and kw['log_time'] == 2 and count_s % 20 == 0:
            print(colored('%r  %2.2f ms per sample on everage' % (method.__name__, (te-t_time) * 1000/count_s), 'green'))
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


def load_json(line, is_debug=False):
    _json = json.loads(line, strict=False)
    return _json['id'], _json['title'], _json['content']


def prep_cut_sent(line, max_sent_words=60, max_sents=90, is_debug=False, log_time=0):
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
