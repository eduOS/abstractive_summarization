# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import glob
from codecs import open
from collections import defaultdict as dd

log_file = open('corpus_log', 'a', 'utf-8')
data_path = "./data/*.txt_*"
filelist = glob.glob(data_path)

len_art = []
len_abs = []
leng_dic = dd(lambda: [])

for ff in filelist:
    with open(ff, 'r', 'utf-8') as f:
        for line in f:
            art, _abs = line.strip().split("\t")
            art_l = len(art.split())
            abs_l = len(_abs.split())
            leng_dic[art_l] += [abs_l]
            if art_l < 20:
                print('art')
                print(art)
            if abs_l < 2:
                print('abs')
                print(_abs)
            len_art.append(art_l)
            len_abs.append(abs_l)


log_file.write("the mean of art: %s" % float(np.mean(len_art)))
log_file.write("\n")
log_file.write("the std of art: %s" % float(np.std(len_art)))
log_file.write("\n")
log_file.write("the max of art: %s" % float(np.max(len_art)))
log_file.write("\n")
log_file.write("the min of art: %s" % float(np.min(len_art)))
log_file.write("\n")
log_file.write("\n")

log_file.write("the mean of abs: %s" % float(np.mean(len_abs)))
log_file.write("\n")
log_file.write("the std of abs: %s" % float(np.std(len_abs)))
log_file.write("\n")
log_file.write("the max of abs: %s" % float(np.max(len_abs)))
log_file.write("\n")
log_file.write("the min of abs: %s" % float(np.min(len_abs)))
log_file.write("\n")
log_file.write("\n")

for l in leng_dic:
    log_file.write("the mean length of abs for art which is of length: %s: %s" % (l, float(np.mean(leng_dic[l]))))
    log_file.write("\n")
    log_file.write("the std length of abs for art which is of length: %s: %s" % (l, float(np.std(leng_dic[l]))))
    log_file.write("\n")
    log_file.write("the max length of abs for art which is of length: %s: %s" % (l, float(np.max(leng_dic[l]))))
    log_file.write("\n")
    log_file.write("the min length of abs for art which is of length: %s: %s" % (l, float(np.min(leng_dic[l]))))
    log_file.write("\n")
    log_file.write("\n")

log_file.close()
