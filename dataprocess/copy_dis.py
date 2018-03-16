# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function
from __future__ import absolute_import
from __future__ import division
import glob
from codecs import open
# from itertools import chain
# import numpy as np
# import matplotlib.pyplot as plt
# from termcolor import colored
import re

tf = open('temp_dis.txt', 'w', 'utf-8')


corpus_files = glob.glob('data/train*')
print(len(corpus_files))
contents = []
references = []
locations = []
lens = []


for corpus_f in corpus_files:
    with open(corpus_f, 'r', 'utf-8') as rf:
        for l in rf:
            content, reference = l.split("\t")
            contents.append(content.strip())
            references.append(reference.strip())

assert len(contents) == len(references)

new_c = []
new_r = []

for _n, (_c, _r) in enumerate(zip(contents, references)):
    _c_l = re.sub(r'[—②①⑤⑥⑧③④。：:∶？；！…?;!|.．～~]', "\n", _c).split("\n")
    __r = _r.split(" ")
    n_cl = []
    for cl in _c_l:
        for r in __r:
            if r in cl:
                cl = re.sub(r"(" + r + r")", r"\x1b[31m\1\x1b[0m", cl)
        n_cl.append(cl)

    _c = "\n".join(n_cl)
    new_c.append(_c)
    new_r.append(_r)

for _c, _r in zip(new_c, new_r):
    print(_r)
    print(_c)
    print("\n")
    input("press enter to continue")
