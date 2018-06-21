# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function
from __future__ import absolute_import
from __future__ import division

import os
from codecs import open

files = next(os.walk('.'))[2]
files = filter(lambda x: len(x) < 3, files)
files = sorted(files, key=lambda x: int(x))
base_file, files = files[0], files[1:]
arti = refe = hypo = None

cs = {}

with open(base_file, "r", 'utf-8') as f:
    for line in f:
        l = line.strip()
        if not l:
            arti = refe = hypo = None
            continue
        if l.startswith('article'):
            arti = l
        if l.startswith('reference'):
            refe = l
        if l.startswith('hypothesis'):
            hypo = l

        if arti and refe and hypo:
            cs[arti] = [refe, [hypo]]

for file in files:
    arti = refe = hypo = None
    with open(file, "r", 'utf-8') as f:
        for line in f:
            l = line.strip()
            if not l:
                arti = refe = hypo = None
                continue
            if l.startswith('article'):
                arti = l
            if l.startswith('hypothesis'):
                hypo = l

            if arti and hypo:
                if arti in cs.keys():
                    cs[arti][1].append(hypo)


with open("comp", "w", 'utf-8') as f:
    for k in cs:
        f.write(k)
        f.write('\n')
        f.write(cs[k][0])
        f.write('\n')

        for h in cs[k][1]:
            f.write("\t" + h)
            f.write('\n')
        f.write('\n')
