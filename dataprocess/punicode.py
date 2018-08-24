# -*- coding: utf-8 -*-
# tested only for python3.5

from __future__ import unicode_literals, print_function
from __future__ import absolute_import
from __future__ import division

from nltk.parse.corenlp import CoreNLPParser
from unidecode import unidecode
# from codecs import open
import json
from termcolor import colored
import re
import unicodedata

stanford = CoreNLPParser()
tk = stanford.tokenize

fi = open('./data/dptest.txt', encoding='unicode_escape')
# fi = open('./data/dptest.txt', encoding='utf-8')


def my_repl(match):
    uni = match.group('uni')
    if uni:
        if unicodedata.category(uni).startswith("P"):
            return colored(' ' + unidecode(uni) + ' ', 'green')
        else:
            return ' [UNK] '


def tokenize(line):
    return ' '.join(tk(line))


def debug(tag, line):
    print(colored(tag + ': ', 'yellow'))
    ipt = input('input the char number to see, f for all chars: ')
    if ipt == "f":
        print(line)
    elif not ipt:
        print(line[:1000])
    elif ipt:
        print(line[:int(ipt)])

while(True):
    line = fi.readline()
    if not line:
        break
    debug('from origin', line)
    # line = line.encode('utf-8', errors='escape')
    # debug('encoded utf8', line)
    line = unidecode(line)
    line = re.sub(pattern=r'(?P<uni>[^\x00-\x7f])', repl=my_repl, string=line)
    debug('unidecoded', line)
    # line = json.dumps(line)
    line = tokenize(line)
    debug('tokenized', line)
    print('\n\n')
