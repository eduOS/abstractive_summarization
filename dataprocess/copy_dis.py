# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function
from __future__ import absolute_import
from __future__ import division
import glob
from codecs import open
from itertools import chain
import numpy as np
import matplotlib.pyplot as plt

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

for _n, (_c, _r) in enumerate(zip(contents, references)):
    _c = _c.split(" ")
    _r = _r.split(" ")
    indices = [n for n, c in enumerate(_c) if c in _r]
    locations.append(indices)
    lens.append(len(_c))
    # print(' '.join(_c))
    # print(" ".join(_r))
    # print(len(list(_c)))
    # print(indices)


positions = np.array(list(chain.from_iterable(locations)))

relative_positions = [list(np.array(li).astype(float)/le) for li, le in zip(locations, lens)]

relative_positions = np.array(list(chain.from_iterable(relative_positions)))
# assert len(relative_positions) == len(positions)
# print(set(list(positions)))
# print(len(set(list(positions))))

plt.hist(relative_positions, bins=50)
plt.savefig('relative_positions.png')

# plt.hist(positions, bins=137)
# plt.hist(positions, bins="auto")
# plt.savefig('positions.png')
